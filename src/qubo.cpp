#include "qubo/qubo.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qubo {

void Qubo::add_pair(int i, int j, double c) {
    if (i == j) {
        h[i] += c;
        return;
    }
    J[static_cast<std::size_t>(i) * n + j] += c;
    J[static_cast<std::size_t>(j) * n + i] += c;
}

double Qubo::energy(const Bits& x) const {
    double e = offset;
    for (int i = 0; i < n; ++i) {
        if (!x[i]) continue;
        e += h[i];
        const double* row = J.data() + static_cast<std::size_t>(i) * n;
        for (int j = i + 1; j < n; ++j) {
            if (x[j]) e += row[j];
        }
    }
    return e;
}

double Ising::energy(const std::vector<int>& s) const {
    double e = offset;
    for (int i = 0; i < n; ++i) {
        e += h[i] * s[i];
        for (int j = i + 1; j < n; ++j) e += J[static_cast<std::size_t>(i) * n + j] * s[i] * s[j];
    }
    return e;
}

Ising to_ising(const Qubo& q) {
    // x_i = (1 + s_i)/2:
    //   h_i x_i        -> h_i/2 + (h_i/2) s_i
    //   J_ij x_i x_j   -> J_ij/4 (1 + s_i + s_j + s_i s_j)
    Ising is;
    is.n = q.n;
    is.h.assign(q.n, 0.0);
    is.J.assign(static_cast<std::size_t>(q.n) * q.n, 0.0);
    is.offset = q.offset;
    for (int i = 0; i < q.n; ++i) {
        is.offset += 0.5 * q.h[i];
        is.h[i] += 0.5 * q.h[i];
        for (int j = i + 1; j < q.n; ++j) {
            const double c = q.coupling(i, j);
            if (c == 0.0) continue;
            is.offset += 0.25 * c;
            is.h[i] += 0.25 * c;
            is.h[j] += 0.25 * c;
            is.J[static_cast<std::size_t>(i) * q.n + j] = 0.25 * c;
            is.J[static_cast<std::size_t>(j) * q.n + i] = 0.25 * c;
        }
    }
    return is;
}

std::vector<int> PortfolioQubo::units(const Bits& x) const {
    std::vector<int> u(n_assets, 0);
    for (int a = 0; a < n_assets; ++a) {
        for (int k = 0; k < spec.bits; ++k) {
            if (x[a * spec.bits + k]) u[a] += 1 << k;
        }
    }
    return u;
}

std::vector<double> PortfolioQubo::weights(const Bits& x) const {
    const auto u = units(x);
    std::vector<double> w(n_assets);
    for (int a = 0; a < n_assets; ++a) w[a] = static_cast<double>(u[a]) / spec.budget_units;
    return w;
}

bool PortfolioQubo::feasible(const Bits& x) const {
    const auto u = units(x);
    int total = 0;
    for (int v : u) total += v;
    return total == spec.budget_units;
}

double PortfolioQubo::objective(const std::vector<double>& w) const {
    double obj = spec.risk_aversion * spec.sigma.quad(w);
    for (int a = 0; a < n_assets; ++a) obj -= spec.mu[a] * w[a];
    for (int a = 0; a < n_assets; ++a) {
        // Empty w_prev = starting from cash, the same convention as the QUBO.
        const double wp = spec.w_prev.empty() ? 0.0 : spec.w_prev[a];
        obj += spec.turnover * (w[a] - wp) * (w[a] - wp);
    }
    return obj;
}

namespace {

// Writes everything except the budget penalty into q.
void add_objective(Qubo& q, const PortfolioSpec& s, int n_assets) {
    const int b = s.bits;
    const double W = s.budget_units;
    auto var = [b](int asset, int bit) { return asset * b + bit; };
    auto val = [](int bit) { return static_cast<double>(1 << bit); };

    // gamma w' Sigma w = gamma / W^2 sum_{p,q} v_p v_q Sigma_{a(p) a(q)} x_p x_q.
    // Each unordered pair appears twice in the double sum, hence the factor 2
    // for p < q; p == q lands on the diagonal and folds into h.
    for (int a = 0; a < n_assets; ++a) {
        for (int k = 0; k < b; ++k) {
            const int p = var(a, k);
            q.add_linear(p, s.risk_aversion * val(k) * val(k) * s.sigma(a, a) / (W * W));
            for (int a2 = 0; a2 < n_assets; ++a2) {
                for (int k2 = 0; k2 < b; ++k2) {
                    const int r = var(a2, k2);
                    if (r <= p) continue;
                    q.add_pair(p, r, 2.0 * s.risk_aversion * val(k) * val(k2) * s.sigma(a, a2) / (W * W));
                }
            }
            q.add_linear(p, -s.mu[a] * val(k) / W);
        }
    }

    // tau sum_a (w_a - wp_a)^2: quadratic part only couples bits of the same asset.
    if (s.turnover != 0.0) {
        for (int a = 0; a < n_assets; ++a) {
            const double wp = s.w_prev.empty() ? 0.0 : s.w_prev[a];
            q.offset += s.turnover * wp * wp;
            for (int k = 0; k < b; ++k) {
                const int p = var(a, k);
                q.add_linear(p, s.turnover * (val(k) * val(k) / (W * W) - 2.0 * wp * val(k) / W));
                for (int k2 = k + 1; k2 < b; ++k2) q.add_pair(p, var(a, k2), 2.0 * s.turnover * val(k) * val(k2) / (W * W));
            }
        }
    }
}

void add_budget_penalty(Qubo& q, int n_assets, int b, int W, double A) {
    // A (sum_p v_p x_p - W)^2 = A [ sum_p v_p^2 x_p + 2 sum_{p<r} v_p v_r x_p x_r - 2 W sum_p v_p x_p + W^2 ]
    const int n = n_assets * b;
    auto val = [b](int p) { return static_cast<double>(1 << (p % b)); };
    for (int p = 0; p < n; ++p) {
        q.add_linear(p, A * (val(p) * val(p) - 2.0 * W * val(p)));
        for (int r = p + 1; r < n; ++r) q.add_pair(p, r, 2.0 * A * val(p) * val(r));
    }
    q.offset += A * static_cast<double>(W) * W;
}

}  // namespace

PortfolioQubo build_portfolio_qubo(const PortfolioSpec& spec) {
    const int n_assets = static_cast<int>(spec.mu.size());
    if (n_assets < 1) throw std::invalid_argument("no assets");
    if (spec.sigma.rows() != static_cast<std::size_t>(n_assets) || spec.sigma.cols() != static_cast<std::size_t>(n_assets)) {
        throw std::invalid_argument("sigma and mu sizes differ");
    }
    if (!spec.w_prev.empty() && spec.w_prev.size() != static_cast<std::size_t>(n_assets)) {
        throw std::invalid_argument("w_prev has the wrong size");
    }
    if (spec.bits < 1 || spec.bits > 8) throw std::invalid_argument("bits per asset must be in [1, 8]");
    const int max_units = n_assets * ((1 << spec.bits) - 1);
    if (spec.budget_units < 1 || spec.budget_units > max_units) {
        throw std::invalid_argument("budget is not reachable with this encoding");
    }

    PortfolioQubo pq{Qubo(n_assets * spec.bits), spec, n_assets, 0.0};
    add_objective(pq.qubo, spec, n_assets);

    double A = spec.penalty;
    if (A < 0.0) {
        // Calibration. Flipping bit p changes the objective by at most
        // B_p = |h_p| + sum_r |J_pr|, while leaving the feasible set costs at
        // least A (the smallest unit is 1, and the penalty grows with the
        // square of the violation). With A above max_p B_p, no single flip
        // out of a feasible point can lower the energy, so every local
        // minimum reachable by one flip from the feasible set is feasible.
        // The 10% margin keeps that strict under rounding. Larger A would be
        // safer still but flattens the landscape the annealers must explore.
        double worst = 0.0;
        for (int p = 0; p < pq.qubo.n; ++p) {
            double bp = std::abs(pq.qubo.h[p]);
            for (int r = 0; r < pq.qubo.n; ++r) bp += std::abs(pq.qubo.coupling(p, r));
            worst = std::max(worst, bp);
        }
        A = 1.1 * worst;
    }
    pq.penalty = A;
    add_budget_penalty(pq.qubo, n_assets, spec.bits, spec.budget_units, A);
    return pq;
}

}  // namespace qubo

// -- laklaa.com
