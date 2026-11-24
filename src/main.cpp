// Command-line front end.
//
//   qubo-alloc demo           one rebalance, every solver against the exact optimum
//   qubo-alloc bench          success probability and time-to-solution vs exact (n <= 24),
//                             and gap vs best known at scale (up to 200 assets)
//   qubo-alloc backtest       ten-year walk-forward, every method, metrics and equity curves
//   qubo-alloc perf           incremental local fields vs naive energy recomputation
//   qubo-alloc export         write one portfolio QUBO in D-Wave's BQM JSON layout
//
// Options are --key value pairs; unknown keys are an error rather than a
// silent default.

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

#include "qubo/backtest.hpp"
#include "qubo/instances.hpp"
#include "qubo/market.hpp"
#include "qubo/rng.hpp"
#include "qubo/solvers.hpp"

using namespace qubo;

namespace {

class Args {
public:
    Args(int argc, char** argv, int first, std::set<std::string> allowed) {
        for (int i = first; i < argc; i += 2) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0) throw std::invalid_argument("expected --option, got '" + k + "'");
            k = k.substr(2);
            if (!allowed.count(k)) throw std::invalid_argument("unknown option --" + k);
            if (i + 1 >= argc) throw std::invalid_argument("--" + k + " needs a value");
            v_[k] = argv[i + 1];
        }
    }
    int i(const std::string& k, int d) const { return v_.count(k) ? std::stoi(v_.at(k)) : d; }
    double f(const std::string& k, double d) const { return v_.count(k) ? std::stod(v_.at(k)) : d; }
    std::string s(const std::string& k, const std::string& d) const { return v_.count(k) ? v_.at(k) : d; }

private:
    std::map<std::string, std::string> v_;
};

struct Named {
    std::string name;
    SolveResult r;
};

std::vector<Named> run_all(const Qubo& q, int sweeps, std::uint64_t seed) {
    std::vector<Named> out;
    out.push_back({"descent", solve_descent(q, 64, seed)});
    out.push_back({"sa", solve_sa(q, {sweeps, 8, -1, -1, seed, false})});
    out.push_back({"sa-pair", solve_sa(q, {sweeps, 8, -1, -1, seed, true})});
    SqaParams sp;
    sp.sweeps = sweeps;
    sp.restarts = 8;
    sp.seed = seed;
    out.push_back({"sqa", solve_sqa(q, sp)});
    PtParams pp;
    pp.sweeps = sweeps;
    pp.restarts = 8;
    pp.seed = seed;
    out.push_back({"pt", solve_pt(q, pp)});
    return out;
}

int cmd_demo(int argc, char** argv) {
    Args a(argc, argv, 2, {"assets", "units", "bits", "sweeps", "seed"});
    const int n = a.i("assets", 24), units = a.i("units", 6), bits = a.i("bits", 1);
    const auto seed = static_cast<std::uint64_t>(a.i("seed", 1));
    const PortfolioSpec spec = random_portfolio_spec(n, units, bits, seed);
    const PortfolioQubo pq = build_portfolio_qubo(spec);
    std::cout << "Portfolio QUBO: " << n << " assets x " << bits << " bit(s) = " << pq.qubo.n << " variables, budget "
              << units << " units, penalty A = " << std::setprecision(4) << pq.penalty << "\n\n";

    std::vector<Named> all;
    if (pq.qubo.n <= 30) all.push_back({"exhaustive", solve_exhaustive(pq.qubo)});
    for (auto& r : run_all(pq.qubo, a.i("sweeps", 1000), seed)) all.push_back(std::move(r));
    const auto w_rr = relax_and_round(spec);
    const double best = std::min_element(all.begin(), all.end(), [](auto& x, auto& y) { return x.r.energy < y.r.energy; })->r.energy;

    std::cout << std::left << std::setw(12) << "solver" << std::right << std::setw(16) << "objective" << std::setw(14)
              << "gap (bp)" << std::setw(11) << "feasible" << std::setw(12) << "time (ms)" << "   holdings\n";
    auto print = [&](const std::string& name, const std::vector<double>& w, bool feas, double energy, double secs) {
        std::string hold;
        for (int k = 0; k < n; ++k) {
            if (w[k] > 0) hold += std::to_string(k) + (bits > 1 ? "x" + std::to_string(static_cast<int>(std::lround(w[k] * units))) : "") + " ";
        }
        std::cout << std::left << std::setw(12) << name << std::right << std::fixed << std::setprecision(6) << std::setw(16)
                  << pq.objective(w) << std::setprecision(2) << std::setw(14)
                  << (energy - best) * 1e4 << std::setw(11) << (feas ? "yes" : "NO") << std::fixed
                  << std::setprecision(1) << std::setw(12) << secs * 1e3 << "   " << hold << "\n";
    };
    for (const auto& r : all) print(r.name, pq.weights(r.r.x), pq.feasible(r.r.x), r.r.energy, r.r.seconds);
    // Energy of the relax-and-round point, for the gap column.
    Bits xr(pq.qubo.n, 0);
    for (int k = 0; k < n; ++k) {
        int u = static_cast<int>(std::lround(w_rr[k] * units));
        for (int b = 0; b < bits; ++b) xr[k * bits + b] = static_cast<std::uint8_t>((u >> b) & 1);
    }
    print("relax-round", w_rr, true, pq.qubo.energy(xr), 0.0);
    return 0;
}

int cmd_bench(int argc, char** argv) {
    Args a(argc, argv, 2, {"instances", "out", "threads"});
    const int instances = a.i("instances", 10);
    const std::filesystem::path out = a.s("out", "results");
    std::filesystem::create_directories(out);
    // One thread: wall time is then CPU time, and time-to-solution figures
    // are comparable between solvers that parallelise differently.
#ifdef _OPENMP
    omp_set_num_threads(a.i("threads", 1));
#endif

    // Part 0: the landscape itself. For one 20-asset, pick-5 instance, the
    // energy of every feasible portfolio (C(20,5) = 15,504) relative to the
    // optimum, next to the height of the penalty wall around each of them.
    {
        const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(20, 5, 1, 900));
        const double opt = solve_exhaustive(pq.qubo).energy;
        std::ofstream fl(out / "landscape.csv");
        fl << "gap_bp,min_wall_bp\n";
        for (unsigned mask = 0; mask < (1u << 20); ++mask) {
            if (std::popcount(mask) != 5) continue;
            Bits x(20);
            for (int i = 0; i < 20; ++i) x[i] = static_cast<std::uint8_t>((mask >> i) & 1);
            FieldState st(pq.qubo, x);
            double wall = 1e300;
            for (int i = 0; i < 20; ++i) wall = std::min(wall, st.delta(i));
            fl << (st.energy() - opt) * 1e4 << "," << wall * 1e4 << "\n";
        }
        std::cout << "Part 0: landscape of one instance written (penalty A = " << pq.penalty * 1e4 << " bp)\n\n";
    }

    // Part 1: exact optimum known. Success = a restart reaching it.
    std::ofstream f1(out / "tts.csv");
    f1 << "assets,solver,sweeps,p_success,seconds_per_run,tts99,mean_gap_bp\n";
    std::cout << "Part 1: time to solution against the exact optimum (pick n/4 of n assets)\n";
    std::cout << std::setw(6) << "assets" << std::setw(8) << "solver" << std::setw(8) << "sweeps" << std::setw(10) << "P(opt)"
              << std::setw(14) << "TTS99 (ms)" << "\n";
    for (int n : {16, 20, 24}) {
        std::vector<PortfolioQubo> pqs;
        std::vector<double> opt;
        for (int k = 0; k < instances; ++k) {
            pqs.push_back(build_portfolio_qubo(random_portfolio_spec(n, n / 4, 1, 1000 * n + k)));
            opt.push_back(solve_exhaustive(pqs.back().qubo).energy);
        }
        for (const std::string solver : {"sa", "sa-pair", "sqa", "pt"}) {
            for (int sweeps : {10, 30, 100, 300, 1000}) {
                int hits = 0, runs = 0;
                double secs = 0.0, gap = 0.0;
                for (int k = 0; k < instances; ++k) {
                    SolveResult r;
                    const int restarts = 16;
                    const auto seed = static_cast<std::uint64_t>(7 + k);
                    if (solver == "sa") r = solve_sa(pqs[k].qubo, {sweeps, restarts, -1, -1, seed, false});
                    if (solver == "sa-pair") r = solve_sa(pqs[k].qubo, {sweeps, restarts, -1, -1, seed, true});
                    if (solver == "sqa") {
                        SqaParams sp;
                        sp.sweeps = sweeps;
                        sp.restarts = restarts;
                        sp.seed = seed;
                        r = solve_sqa(pqs[k].qubo, sp);
                    }
                    if (solver == "pt") {
                        PtParams pp;
                        pp.sweeps = sweeps;
                        pp.restarts = restarts;
                        pp.seed = seed;
                        r = solve_pt(pqs[k].qubo, pp);
                    }
                    const double tol = 1e-9 * (1.0 + std::abs(opt[k]));
                    for (double e : r.restart_energies) {
                        ++runs;
                        if (e <= opt[k] + tol) ++hits;
                        gap += (e - opt[k]) * 1e4;  // basis points of annual utility
                    }
                    secs += r.seconds / restarts;
                }
                const double p = static_cast<double>(hits) / runs;
                const double t_run = secs / instances;
                const double tts = time_to_solution(t_run, p);
                f1 << n << "," << solver << "," << sweeps << "," << p << "," << t_run << "," << tts << "," << gap / runs << "\n";
                std::cout << std::setw(6) << n << std::setw(8) << solver << std::setw(8) << sweeps << std::setw(10)
                          << std::fixed << std::setprecision(3) << p << std::setw(14) << std::setprecision(3)
                          << tts * 1e3 << "\n";
            }
        }
    }

    // Part 2: no exact answer. Reference = best energy found by any solver
    // with a large budget; we report the gap of smaller budgets to it.
    std::ofstream f2(out / "scaling.csv");
    f2 << "assets,solver,sweeps,mean_gap_bp,seconds\n";
    std::cout << "\nPart 2: gap to best known, in basis points of annual utility (pick n/5 of n assets, 5 instances)\n";
    for (int n : {50, 100, 200}) {
        std::vector<PortfolioQubo> pqs;
        std::vector<double> ref;
        for (int k = 0; k < 5; ++k) {
            pqs.push_back(build_portfolio_qubo(random_portfolio_spec(n, n / 5, 1, 5000 * n + k)));
            double b = 1e300;
            for (auto& r : run_all(pqs.back().qubo, 3000, 99)) b = std::min(b, r.r.energy);
            ref.push_back(b);
        }
        for (const std::string solver : {"descent", "sa", "sa-pair", "sqa", "pt"}) {
            for (int sweeps : {10, 30, 100, 300, 1000}) {
                if (solver == "descent" && sweeps != 10) continue;
                double gap = 0.0, secs = 0.0;
                for (int k = 0; k < 5; ++k) {
                    SolveResult r;
                    const auto seed = static_cast<std::uint64_t>(11 + k);
                    if (solver == "descent") r = solve_descent(pqs[k].qubo, 64, seed);
                    if (solver == "sa") r = solve_sa(pqs[k].qubo, {sweeps, 4, -1, -1, seed, false});
                    if (solver == "sa-pair") r = solve_sa(pqs[k].qubo, {sweeps, 4, -1, -1, seed, true});
                    if (solver == "sqa") {
                        SqaParams sp;
                        sp.sweeps = sweeps;
                        sp.restarts = 4;
                        sp.seed = seed;
                        r = solve_sqa(pqs[k].qubo, sp);
                    }
                    if (solver == "pt") {
                        PtParams pp;
                        pp.sweeps = sweeps;
                        pp.restarts = 1;
                        pp.seed = seed;
                        r = solve_pt(pqs[k].qubo, pp);
                    }
                    gap += (r.energy - ref[k]) * 1e4;
                    secs += r.seconds;
                }
                f2 << n << "," << solver << "," << sweeps << "," << gap / 5 << "," << secs / 5 << "\n";
                std::cout << std::setw(6) << n << std::setw(9) << solver << std::setw(7) << sweeps << "  gap "
                          << std::fixed << std::setprecision(2) << std::setw(9) << gap / 5 << " bp  time " << std::fixed
                          << std::setprecision(1) << secs / 5 * 1e3 << " ms\n";
            }
        }
    }
    std::cout << "\nwritten to " << out << "\n";
    return 0;
}

struct PathStats {
    std::vector<double> sharpe, cagr, vol, mdd, turnover, gap_bp, at_best, solve_s;
};

double mean_of(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }
double se_of(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean_of(v);
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(v.size() - 1) / static_cast<double>(v.size()));
}

int cmd_backtest(int argc, char** argv) {
    Args a(argc, argv, 2, {"assets", "days", "units", "bits", "sweeps", "cost-bps", "risk-aversion", "turnover", "out", "data",
                           "prices", "methods", "seed", "paths"});
    BacktestConfig cfg;
    cfg.units = a.i("units", 8);
    cfg.bits = a.i("bits", 1);
    cfg.sweeps = a.i("sweeps", 400);
    cfg.cost_bps = a.f("cost-bps", 10.0);
    cfg.risk_aversion = a.f("risk-aversion", 4.0);
    cfg.turnover = a.f("turnover", 0.05);
    const std::filesystem::path out = a.s("out", "results");
    std::filesystem::create_directories(out);
    const bool real_data = !a.s("data", "").empty();
    // One simulated path is one draw of history; strategy differences of a
    // few Sharpe points are noise on a single path. With --paths K the whole
    // backtest is repeated on K independent markets.
    const int paths = real_data ? 1 : a.i("paths", 1);

    std::vector<Method> methods;
    {
        std::string list = a.s("methods", "equal-weight,relax-round,descent,sa,sa-pair,sqa,pt");
        std::size_t pos = 0;
        while (pos <= list.size()) {
            const auto next = list.find(',', pos);
            methods.push_back(method_from_name(list.substr(pos, next - pos)));
            if (next == std::string::npos) break;
            pos = next + 1;
        }
    }

    std::vector<PathStats> stats(methods.size());
    std::ofstream fp(out / "backtest_paths.csv");
    fp << "path,method,cagr,vol,sharpe,max_dd,turnover_per_year,mean_obj_gap_bp,dates_at_best,solver_seconds,repaired\n";
    for (int path = 0; path < paths; ++path) {
        Market m;
        if (real_data) {
            m = load_csv(a.s("data", ""), a.i("prices", 0) != 0);
        } else {
            MarketConfig mc;
            mc.n_assets = a.i("assets", 40);
            mc.n_days = a.i("days", 2520);
            mc.seed = static_cast<std::uint64_t>(a.i("seed", 7) + path);
            m = simulate_market(mc);
        }
        if (path == 0) {
            std::cout << m.returns.cols() << " assets, " << m.returns.rows() << " days, hold " << cfg.units
                      << " units, rebalance every " << cfg.rebalance << " days, costs " << cfg.cost_bps << " bps, " << paths
                      << " path(s)\n";
        }
        std::vector<BacktestResult> results;
        for (Method meth : methods) results.push_back(run_backtest(m, meth, cfg));

        // Per-date gap of each method's objective to the best method that date.
        const std::size_t dates = results.front().objective.size();
        std::vector<double> best(dates, 1e300);
        for (const auto& r : results) {
            if (r.method == Method::EqualWeight) continue;  // not solving the problem
            for (std::size_t d = 0; d < dates; ++d) best[d] = std::min(best[d], r.objective[d]);
        }
        for (std::size_t k = 0; k < results.size(); ++k) {
            const auto& r = results[k];
            double gap = 0.0;
            int at_best = 0;
            for (std::size_t d = 0; d < dates; ++d) {
                const double g = r.objective[d] - best[d];
                gap += g;
                if (g <= 1e-10) ++at_best;
            }
            const double gap_bp = gap / static_cast<double>(dates) * 1e4;
            const bool solves = r.method != Method::EqualWeight;
            const auto& mt = r.metrics;
            auto& st = stats[k];
            st.sharpe.push_back(mt.sharpe);
            st.cagr.push_back(mt.cagr);
            st.vol.push_back(mt.ann_vol);
            st.mdd.push_back(mt.max_drawdown);
            st.turnover.push_back(mt.turnover_per_year);
            st.gap_bp.push_back(solves ? gap_bp : NAN);
            st.at_best.push_back(solves ? static_cast<double>(at_best) / static_cast<double>(dates) : NAN);
            st.solve_s.push_back(r.solver_seconds);
            fp << path << "," << method_name(r.method) << "," << mt.cagr << "," << mt.ann_vol << "," << mt.sharpe << ","
               << mt.max_drawdown << "," << mt.turnover_per_year << "," << (solves ? gap_bp : NAN) << ","
               << (solves ? at_best : -1) << "," << r.solver_seconds << "," << r.infeasible_repaired << "\n";
        }
        if (path == 0) {
            std::ofstream fe(out / "equity.csv");
            fe << "day";
            for (const auto& r : results) fe << "," << method_name(r.method);
            fe << "\n";
            for (std::size_t t = 0; t < results.front().equity.size(); ++t) {
                fe << t;
                for (const auto& r : results) fe << "," << r.equity[t];
                fe << "\n";
            }
        }
    }

    std::ofstream fm(out / "backtest_metrics.csv");
    fm << "method,sharpe,sharpe_se,cagr,vol,max_dd,turnover_per_year,obj_gap_bp,obj_gap_bp_se,share_at_best,solver_seconds\n";
    std::cout << "\n" << std::left << std::setw(14) << "method" << std::right << std::setw(16) << "Sharpe" << std::setw(9)
              << "CAGR" << std::setw(8) << "vol" << std::setw(8) << "maxDD" << std::setw(9) << "turn/yr" << std::setw(18)
              << "obj gap (bp)" << std::setw(9) << "at best" << std::setw(9) << "solve s" << "\n";
    for (std::size_t k = 0; k < methods.size(); ++k) {
        const auto& st = stats[k];
        const bool solves = methods[k] != Method::EqualWeight;
        fm << method_name(methods[k]) << "," << mean_of(st.sharpe) << "," << se_of(st.sharpe) << "," << mean_of(st.cagr) << ","
           << mean_of(st.vol) << "," << mean_of(st.mdd) << "," << mean_of(st.turnover) << ","
           << (solves ? mean_of(st.gap_bp) : NAN) << "," << (solves ? se_of(st.gap_bp) : NAN) << ","
           << (solves ? mean_of(st.at_best) : NAN) << "," << mean_of(st.solve_s) << "\n";
        std::cout << std::left << std::setw(14) << method_name(methods[k]) << std::right << std::fixed << std::setprecision(3)
                  << std::setw(8) << mean_of(st.sharpe) << " +/- " << std::setw(5) << se_of(st.sharpe) << std::setprecision(2)
                  << std::setw(7) << 100 * mean_of(st.cagr) << "%" << std::setw(7) << 100 * mean_of(st.vol) << "%"
                  << std::setw(7) << 100 * mean_of(st.mdd) << "%" << std::setw(9) << mean_of(st.turnover);
        if (solves) {
            std::cout << std::setw(10) << mean_of(st.gap_bp) << " +/- " << std::setw(4) << std::setprecision(1)
                      << se_of(st.gap_bp) << std::setw(8) << std::setprecision(0) << 100 * mean_of(st.at_best) << "%";
        } else {
            std::cout << std::setw(18) << "-" << std::setw(9) << "-";
        }
        std::cout << std::setprecision(2) << std::setw(9) << mean_of(st.solve_s) << "\n";
    }
    std::cout << "\nwritten to " << out << "\n";
    return 0;
}

int cmd_perf(int argc, char** argv) {
    Args a(argc, argv, 2, {"out"});
    const std::filesystem::path out = a.s("out", "results");
    std::filesystem::create_directories(out);
    std::ofstream f(out / "perf.csv");
    f << "variables,incremental_ns_per_proposal,naive_ns_per_proposal\n";
    std::cout << std::setw(10) << "variables" << std::setw(18) << "incremental ns" << std::setw(14) << "naive ns" << std::setw(10)
              << "speedup\n";
    for (int n : {25, 50, 100, 200, 400}) {
        const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(n, n / 5, 1, 77));
        const Qubo& q = pq.qubo;
        Rng rng(1);
        Bits x(n);
        for (auto& b : x) b = static_cast<std::uint8_t>(rng.below(2));
        const double beta = 1.0 / estimate_energy_scale(q, 1).median;

        // Incremental: O(1) proposal, O(n) on acceptance.
        FieldState st(q, x);
        const long long props_inc = 20'000'000LL / n * 10;
        auto t0 = std::chrono::steady_clock::now();
        for (long long s = 0; s < props_inc / n; ++s) {
            for (int i = 0; i < n; ++i) {
                const double d = st.delta(i);
                if (d <= 0.0 || rng.uniform() < std::exp(-beta * d)) st.flip(i);
            }
        }
        const double inc = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
                           static_cast<double>(props_inc / n * n);

        // Naive: recompute the full energy for each proposal.
        Bits y = x;
        double e = q.energy(y);
        const long long props_naive = std::max(2000LL, 4'000'000LL / (static_cast<long long>(n) * n) * 50);
        t0 = std::chrono::steady_clock::now();
        for (long long s = 0; s < props_naive; ++s) {
            const int i = static_cast<int>(s % n);
            y[i] ^= 1;
            const double e2 = q.energy(y);
            const double d = e2 - e;
            if (d <= 0.0 || rng.uniform() < std::exp(-beta * d)) {
                e = e2;
            } else {
                y[i] ^= 1;
            }
        }
        const double naive = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
                             static_cast<double>(props_naive);
        f << n << "," << inc << "," << naive << "\n";
        std::cout << std::setw(10) << n << std::setw(18) << std::fixed << std::setprecision(1) << inc << std::setw(14) << naive
                  << std::setw(9) << std::setprecision(0) << naive / inc << "x\n";
    }
    return 0;
}

int cmd_export(int argc, char** argv) {
    Args a(argc, argv, 2, {"assets", "units", "bits", "seed", "out"});
    const PortfolioQubo pq = build_portfolio_qubo(
        random_portfolio_spec(a.i("assets", 20), a.i("units", 5), a.i("bits", 1), static_cast<std::uint64_t>(a.i("seed", 1))));
    const std::string path = a.s("out", "portfolio_bqm.json");
    std::ofstream f(path);
    f.precision(17);
    // Same layout as dimod's BinaryQuadraticModel.to_serializable(), minus
    // the version metadata: linear and quadratic terms plus the offset.
    f << "{\n  \"vartype\": \"BINARY\",\n  \"offset\": " << pq.qubo.offset << ",\n  \"linear\": {";
    for (int i = 0; i < pq.qubo.n; ++i) f << (i ? ", " : "") << "\"" << i << "\": " << pq.qubo.h[i];
    f << "},\n  \"quadratic\": [";
    bool first = true;
    for (int i = 0; i < pq.qubo.n; ++i) {
        for (int j = i + 1; j < pq.qubo.n; ++j) {
            const double c = pq.qubo.coupling(i, j);
            if (c == 0.0) continue;
            f << (first ? "" : ", ") << "[" << i << ", " << j << ", " << c << "]";
            first = false;
        }
    }
    f << "]\n}\n";
    std::cerr << "wrote " << path << " (" << pq.qubo.n << " variables)\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: qubo-alloc <demo|bench|backtest|perf|export> [--option value ...]\n";
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "demo") return cmd_demo(argc, argv);
        if (cmd == "bench") return cmd_bench(argc, argv);
        if (cmd == "backtest") return cmd_backtest(argc, argv);
        if (cmd == "perf") return cmd_perf(argc, argv);
        if (cmd == "export") return cmd_export(argc, argv);
        std::cerr << "unknown command '" << cmd << "'\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}

// -- laklaa.com
