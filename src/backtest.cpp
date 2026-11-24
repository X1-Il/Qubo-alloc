#include "qubo/backtest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "qubo/estimators.hpp"
#include "qubo/solvers.hpp"

namespace qubo {

std::string method_name(Method m) {
    switch (m) {
        case Method::EqualWeight: return "equal-weight";
        case Method::RelaxRound: return "relax-round";
        case Method::Descent: return "descent";
        case Method::SA: return "sa";
        case Method::SAPair: return "sa-pair";
        case Method::SQA: return "sqa";
        case Method::PT: return "pt";
        case Method::Exhaustive: return "exhaustive";
    }
    return "?";
}

Method method_from_name(const std::string& s) {
    for (Method m : {Method::EqualWeight, Method::RelaxRound, Method::Descent, Method::SA, Method::SAPair, Method::SQA, Method::PT,
                     Method::Exhaustive}) {
        if (method_name(m) == s) return m;
    }
    throw std::invalid_argument("unknown method '" + s + "'");
}

std::vector<double> project_simplex(const std::vector<double>& v) {
    // Sort descending, find the largest rho with u_rho - (sum_{j<=rho} u_j - 1)/rho > 0,
    // then shift everything by that threshold and clip at zero.
    std::vector<double> u(v);
    std::sort(u.begin(), u.end(), std::greater<>());
    double cum = 0.0, theta = 0.0;
    for (std::size_t j = 0; j < u.size(); ++j) {
        cum += u[j];
        const double t = (cum - 1.0) / static_cast<double>(j + 1);
        if (u[j] - t > 0.0) theta = t;
    }
    std::vector<double> w(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) w[i] = std::max(v[i] - theta, 0.0);
    return w;
}

namespace {

double objective_of(const PortfolioSpec& s, const std::vector<double>& w) {
    double obj = s.risk_aversion * s.sigma.quad(w);
    for (std::size_t a = 0; a < w.size(); ++a) {
        // An empty w_prev means trading out of cash, exactly as in the QUBO.
        const double wp = s.w_prev.empty() ? 0.0 : s.w_prev[a];
        obj -= s.mu[a] * w[a];
        obj += s.turnover * (w[a] - wp) * (w[a] - wp);
    }
    return obj;
}

std::vector<double> to_weights(const std::vector<int>& units, int W) {
    std::vector<double> w(units.size());
    for (std::size_t a = 0; a < units.size(); ++a) w[a] = static_cast<double>(units[a]) / W;
    return w;
}

}  // namespace

std::vector<double> relax_and_round(const PortfolioSpec& s) {
    const std::size_t n = s.mu.size();
    // f(w) = gamma w' S w - mu' w + tau ||w - wp||^2, gradient Lipschitz with
    // L = 2 gamma lambda_max(S) + 2 tau. FISTA reaches 1e-8 in a few hundred
    // iterations at this size; the rounding error dwarfs that anyway.
    const double L = 2.0 * s.risk_aversion * s.sigma.max_eigenvalue() + 2.0 * s.turnover;
    std::vector<double> w(n, 1.0 / static_cast<double>(n)), y = w, w_old = w;
    double tk = 1.0;
    for (int it = 0; it < 500; ++it) {
        auto g = s.sigma.mul(y);
        std::vector<double> step(n);
        for (std::size_t a = 0; a < n; ++a) {
            const double wp = s.w_prev.empty() ? 0.0 : s.w_prev[a];
            const double grad = 2.0 * s.risk_aversion * g[a] - s.mu[a] + 2.0 * s.turnover * (y[a] - wp);
            step[a] = y[a] - grad / L;
        }
        w_old = w;
        w = project_simplex(step);
        const double tn = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * tk * tk));
        for (std::size_t a = 0; a < n; ++a) y[a] = w[a] + ((tk - 1.0) / tn) * (w[a] - w_old[a]);
        tk = tn;
    }

    // Largest-remainder rounding onto units in [0, cap], summing to W.
    const int W = s.budget_units;
    const int cap = (1 << s.bits) - 1;
    std::vector<int> units(n, 0);
    std::vector<std::pair<double, std::size_t>> rem;
    int used = 0;
    for (std::size_t a = 0; a < n; ++a) {
        const double target = w[a] * W;
        units[a] = std::min(cap, static_cast<int>(std::floor(target)));
        used += units[a];
        rem.emplace_back(target - units[a], a);
    }
    std::sort(rem.begin(), rem.end(), [](const auto& l, const auto& r) { return l.first > r.first || (l.first == r.first && l.second < r.second); });
    for (std::size_t k = 0; used < W && k < rem.size(); ++k) {
        if (units[rem[k].second] < cap) {
            ++units[rem[k].second];
            ++used;
        }
    }
    // If capping left units over, spread them onto any asset with room.
    for (std::size_t a = 0; used < W && a < n; ++a) {
        while (used < W && units[a] < cap) {
            ++units[a];
            ++used;
        }
    }
    return to_weights(units, W);
}

std::vector<double> repair_to_budget(const PortfolioQubo& pq, std::vector<int> units) {
    const auto& s = pq.spec;
    const int cap = (1 << s.bits) - 1;
    auto total = [&] { return std::accumulate(units.begin(), units.end(), 0); };
    while (total() != s.budget_units) {
        const int dir = total() > s.budget_units ? -1 : 1;
        int best_a = -1;
        double best_obj = 0.0;
        for (int a = 0; a < pq.n_assets; ++a) {
            const int nu = units[a] + dir;
            if (nu < 0 || nu > cap) continue;
            units[a] = nu;
            const double obj = objective_of(s, to_weights(units, s.budget_units));
            units[a] -= dir;
            if (best_a < 0 || obj < best_obj) {
                best_a = a;
                best_obj = obj;
            }
        }
        if (best_a < 0) throw std::logic_error("budget unreachable during repair");
        units[best_a] += dir;
    }
    return to_weights(units, s.budget_units);
}

PortfolioSpec spec_at(const Market& m, int t, const BacktestConfig& cfg, const std::vector<double>& w_prev) {
    if (t < cfg.window) throw std::invalid_argument("not enough history before t");
    const std::size_t n = m.returns.cols();
    Matrix win(static_cast<std::size_t>(cfg.window), n);
    for (int k = 0; k < cfg.window; ++k) {
        const double* src = m.returns.row(static_cast<std::size_t>(t - cfg.window + k));
        std::copy(src, src + n, win.row(static_cast<std::size_t>(k)));
    }
    PortfolioSpec s;
    s.sigma = ledoit_wolf(win).sigma;
    s.sigma *= 252.0;
    s.mu = james_stein_means(win);
    for (auto& v : s.mu) v *= 252.0;
    s.w_prev = w_prev;
    s.risk_aversion = cfg.risk_aversion;
    s.turnover = cfg.turnover;
    s.bits = cfg.bits;
    s.budget_units = cfg.units;
    return s;
}

std::vector<double> decide_weights(const Market& m, int t, Method method, const BacktestConfig& cfg,
                                   const std::vector<double>& w_prev, double* solver_seconds, bool* repaired) {
    const std::size_t n = m.returns.cols();
    if (repaired) *repaired = false;
    if (method == Method::EqualWeight) return std::vector<double>(n, 1.0 / static_cast<double>(n));

    const PortfolioSpec spec = spec_at(m, t, cfg, w_prev);
    const auto start = std::chrono::steady_clock::now();
    std::vector<double> w;
    if (method == Method::RelaxRound) {
        w = relax_and_round(spec);
    } else {
        const PortfolioQubo pq = build_portfolio_qubo(spec);
        // Seed varies with the date so that runs are reproducible but not
        // correlated across rebalances.
        const std::uint64_t seed = cfg.seed * 1000003ULL + static_cast<std::uint64_t>(t);
        SolveResult r;
        switch (method) {
            case Method::Descent: r = solve_descent(pq.qubo, cfg.restarts * 4, seed); break;
            case Method::SA: r = solve_sa(pq.qubo, {cfg.sweeps, cfg.restarts, -1, -1, seed, false}); break;
            case Method::SAPair: r = solve_sa(pq.qubo, {cfg.sweeps, cfg.restarts, -1, -1, seed, true}); break;
            case Method::SQA: {
                SqaParams sp;
                sp.sweeps = cfg.sweeps;
                sp.restarts = cfg.restarts;
                sp.seed = seed;
                r = solve_sqa(pq.qubo, sp);
                break;
            }
            case Method::PT: {
                PtParams pp;
                pp.sweeps = cfg.sweeps;
                pp.restarts = std::max(1, cfg.restarts / 4);
                pp.seed = seed;
                r = solve_pt(pq.qubo, pp);
                break;
            }
            case Method::Exhaustive: r = solve_exhaustive(pq.qubo); break;
            default: throw std::logic_error("unreachable");
        }
        if (pq.feasible(r.x)) {
            w = pq.weights(r.x);
        } else {
            w = repair_to_budget(pq, pq.units(r.x));
            if (repaired) *repaired = true;
        }
    }
    if (solver_seconds) *solver_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return w;
}

Metrics compute_metrics(const std::vector<double>& daily, const std::vector<double>& turnover,
                        const std::vector<double>& names_held) {
    Metrics mt;
    if (daily.empty()) return mt;
    const double nd = static_cast<double>(daily.size());
    double mean = 0.0;
    for (double r : daily) mean += r;
    mean /= nd;
    double var = 0.0;
    for (double r : daily) var += (r - mean) * (r - mean);
    var /= std::max(1.0, nd - 1.0);
    double eq = 1.0, peak = 1.0, mdd = 0.0;
    for (double r : daily) {
        eq *= 1.0 + r;
        peak = std::max(peak, eq);
        mdd = std::max(mdd, 1.0 - eq / peak);
    }
    const double years = nd / 252.0;
    mt.cagr = std::pow(eq, 1.0 / years) - 1.0;
    mt.ann_vol = std::sqrt(var * 252.0);
    mt.sharpe = var > 0.0 ? mean / std::sqrt(var) * std::sqrt(252.0) : 0.0;
    mt.max_drawdown = mdd;
    mt.turnover_per_year = std::accumulate(turnover.begin(), turnover.end(), 0.0) / years;
    mt.avg_names = names_held.empty() ? 0.0 : std::accumulate(names_held.begin(), names_held.end(), 0.0) / static_cast<double>(names_held.size());
    return mt;
}

BacktestResult run_backtest(const Market& m, Method method, const BacktestConfig& cfg) {
    const int T = static_cast<int>(m.returns.rows());
    const std::size_t n = m.returns.cols();
    if (T <= cfg.window + 1) throw std::invalid_argument("not enough data for the estimation window");

    BacktestResult res;
    res.method = method;
    std::vector<double> w(n, 0.0);  // start in cash
    std::vector<double> names;
    double equity = 1.0;
    res.equity.push_back(equity);
    for (int t = cfg.window; t < T; ++t) {
        double cost = 0.0;
        if ((t - cfg.window) % cfg.rebalance == 0) {
            bool repaired = false;
            // For the very first trade there is no position to stay close to.
            const std::vector<double> w_prev = (t == cfg.window) ? std::vector<double>{} : w;
            auto target = decide_weights(m, t, method, cfg, w_prev, &res.solver_seconds, &repaired);
            if (repaired) ++res.infeasible_repaired;
            double traded = 0.0;
            for (std::size_t a = 0; a < n; ++a) traded += std::abs(target[a] - w[a]);
            cost = traded * cfg.cost_bps * 1e-4;
            res.turnover.push_back(traded);
            res.rebalance_days.push_back(t);
            // Record the same penalty-free objective for every method, so
            // solver quality can be compared date by date.
            res.objective.push_back(objective_of(spec_at(m, t, cfg, w_prev), target));
            names.push_back(static_cast<double>(std::count_if(target.begin(), target.end(), [](double x) { return x > 1e-12; })));
            w = std::move(target);
        }
        double gross = 0.0;
        std::vector<double> grown(n);
        for (std::size_t a = 0; a < n; ++a) {
            const double r = std::expm1(m.returns(static_cast<std::size_t>(t), a));
            grown[a] = w[a] * (1.0 + r);
            gross += w[a] * r;
        }
        // Weights drift with prices between rebalances.
        const double total = 1.0 + gross;
        for (std::size_t a = 0; a < n; ++a) w[a] = grown[a] / total;
        const double net = gross - cost;
        res.daily.push_back(net);
        equity *= 1.0 + net;
        res.equity.push_back(equity);
    }
    res.metrics = compute_metrics(res.daily, res.turnover, names);
    return res;
}

}  // namespace qubo

// -- laklaa.com
