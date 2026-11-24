// Model-level tests: QUBO algebra, Ising mapping, portfolio formulation,
// estimators, market simulator.

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "qubo/estimators.hpp"
#include "qubo/instances.hpp"
#include "qubo/market.hpp"
#include "qubo/qubo.hpp"
#include "qubo/rng.hpp"
#include "qubo/solvers.hpp"
#include "test.hpp"

using namespace qubo;

namespace {
Qubo random_qubo(int n, std::uint64_t seed) {
    Rng rng(seed);
    Qubo q(n);
    q.offset = rng.normal();
    for (int i = 0; i < n; ++i) {
        q.add_linear(i, rng.normal());
        for (int j = i + 1; j < n; ++j) q.add_pair(i, j, rng.normal());
    }
    return q;
}

Bits random_bits(int n, Rng& rng) {
    Bits x(n);
    for (auto& b : x) b = static_cast<std::uint8_t>(rng.below(2));
    return x;
}
}  // namespace

TEST(field_state_matches_full_energy_after_many_flips) {
    const Qubo q = random_qubo(40, 1);
    Rng rng(2);
    FieldState st(q, random_bits(40, rng));
    CHECK_NEAR(st.energy(), q.energy(st.x()), 1e-10);
    for (int step = 0; step < 5000; ++step) {
        const int i = static_cast<int>(rng.below(40));
        Bits y = st.x();
        y[i] ^= 1;
        const double predicted = st.energy() + st.delta(i);
        st.flip(i);
        CHECK_NEAR(st.energy(), predicted, 1e-9);
        if (step % 500 == 0) CHECK_NEAR(st.energy(), q.energy(y), 1e-8);
    }
    // Fields must equal h + J x exactly as recomputed from scratch.
    FieldState fresh(q, st.x());
    for (int i = 0; i < 40; ++i) CHECK_NEAR(st.fields()[i], fresh.fields()[i], 1e-9);
}

TEST(ising_mapping_preserves_every_energy) {
    const Qubo q = random_qubo(10, 3);
    const Ising is = to_ising(q);
    for (int mask = 0; mask < (1 << 10); ++mask) {
        Bits x(10);
        std::vector<int> s(10);
        for (int i = 0; i < 10; ++i) {
            x[i] = static_cast<std::uint8_t>((mask >> i) & 1);
            s[i] = x[i] ? 1 : -1;
        }
        CHECK_NEAR(q.energy(x), is.energy(s), 1e-10);
    }
}

TEST(portfolio_qubo_energy_equals_objective_plus_penalty) {
    // For every bit pattern of a small multi-bit problem, the QUBO energy must
    // equal the financial objective computed from weights plus A (sum u - W)^2.
    for (int bits : {1, 2, 3}) {
        PortfolioSpec s = random_portfolio_spec(4, bits == 1 ? 3 : 5, bits, 100 + bits);
        const PortfolioQubo pq = build_portfolio_qubo(s);
        const int n = pq.qubo.n;
        for (int mask = 0; mask < (1 << n); ++mask) {
            Bits x(n);
            for (int i = 0; i < n; ++i) x[i] = static_cast<std::uint8_t>((mask >> i) & 1);
            const auto u = pq.units(x);
            int total = 0;
            for (int v : u) total += v;
            const double expected = pq.objective(pq.weights(x)) + pq.penalty * (total - s.budget_units) * (total - s.budget_units);
            CHECK_NEAR(pq.qubo.energy(x), expected, 1e-9 * (1.0 + std::abs(expected)));
        }
    }
}

TEST(penalty_makes_the_optimum_feasible) {
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(18, 6, 1, seed));
        const SolveResult r = solve_exhaustive(pq.qubo);
        CHECK_MSG(pq.feasible(r.x), "seed " << seed);
        // Single flips out of the optimum must all go uphill: it is a strict
        // local minimum, as the calibration argument promises.
        FieldState st(pq.qubo, r.x);
        for (int i = 0; i < pq.qubo.n; ++i) CHECK(st.delta(i) > 0.0);
    }
}

TEST(builder_rejects_inconsistent_inputs) {
    PortfolioSpec s = random_portfolio_spec(5, 3, 1, 9);
    s.budget_units = 6;  // 5 assets x 1 bit can hold at most 5 units
    bool threw = false;
    try {
        build_portfolio_qubo(s);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(ledoit_wolf_beats_sample_covariance_when_data_is_short) {
    // Known covariance: one factor plus noise. With T barely above N the
    // sample estimate is badly conditioned; shrinkage must be closer in
    // Frobenius norm, have a sensible intensity, and stay positive definite.
    const int n = 40, t_len = 60;
    Rng rng(4);
    std::vector<double> beta(n);
    for (auto& b : beta) b = 0.5 + rng.uniform();
    Matrix truth(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) truth(i, j) = beta[i] * beta[j] + (i == j ? 1.0 : 0.0);
    double sample_err = 0.0, lw_err = 0.0;
    for (int rep = 0; rep < 20; ++rep) {
        Matrix X(t_len, n);
        for (int t = 0; t < t_len; ++t) {
            const double f = rng.normal();
            for (int i = 0; i < n; ++i) X(t, i) = beta[i] * f + rng.normal();
        }
        const Shrinkage lw = ledoit_wolf(X);
        CHECK(lw.intensity > 0.0 && lw.intensity < 1.0);
        sample_err += sample_covariance(X).frobenius_distance(truth);
        lw_err += lw.sigma.frobenius_distance(truth);
        // Positive definite: x' S x > 0 for random x.
        std::vector<double> v(n);
        for (auto& e : v) e = rng.normal();
        CHECK(lw.sigma.quad(v) > 0.0);
    }
    CHECK_MSG(lw_err < sample_err, "LW " << lw_err << " vs sample " << sample_err);
}

TEST(ledoit_wolf_with_lots_of_data_barely_shrinks) {
    Rng rng(5);
    Matrix X(20000, 5);
    for (std::size_t t = 0; t < X.rows(); ++t)
        for (std::size_t j = 0; j < 5; ++j) X(t, j) = rng.normal() * (1.0 + static_cast<double>(j));
    CHECK(ledoit_wolf(X).intensity < 0.01);
}

TEST(james_stein_shrinks_toward_grand_mean) {
    Rng rng(6);
    Matrix X(50, 30);
    for (std::size_t t = 0; t < 50; ++t)
        for (std::size_t j = 0; j < 30; ++j) X(t, j) = 0.001 + 0.02 * rng.normal();
    const auto raw = column_means(X);
    const auto js = james_stein_means(X);
    double grand = 0.0;
    for (double v : raw) grand += v;
    grand /= 30.0;
    double sr = 0.0, sj = 0.0;
    for (int j = 0; j < 30; ++j) {
        sr += (raw[j] - grand) * (raw[j] - grand);
        sj += (js[j] - grand) * (js[j] - grand);
    }
    CHECK(sj < sr);
    // Pure noise around a common mean: the estimator should shrink hard.
    CHECK(sj < 0.25 * sr);
}

TEST(simulated_market_has_the_requested_volatility_and_fat_tails) {
    MarketConfig mc;
    mc.n_assets = 20;
    mc.n_days = 252 * 20;
    const Market m = simulate_market(mc);
    // Equal-weight index: dominated by the market factor, so its vol should
    // sit near beta_avg * 16% plus a little diversified noise.
    std::vector<double> idx(m.returns.rows(), 0.0);
    for (std::size_t t = 0; t < m.returns.rows(); ++t)
        for (std::size_t j = 0; j < m.returns.cols(); ++j) idx[t] += m.returns(t, j) / 20.0;
    double mean = 0.0;
    for (double r : idx) mean += r;
    const double nd = static_cast<double>(idx.size());
    mean /= nd;
    double var = 0.0, m4 = 0.0;
    for (double r : idx) {
        var += (r - mean) * (r - mean);
        m4 += std::pow(r - mean, 4);
    }
    var /= nd;
    m4 /= nd;
    const double vol = std::sqrt(var * 252.0);
    const double kurt = m4 / (var * var);
    CHECK_MSG(vol > 0.12 && vol < 0.25, "vol " << vol);
    CHECK_MSG(kurt > 3.5, "excess kurtosis expected, got " << kurt);
}

TEST(csv_round_trip) {
    MarketConfig mc;
    mc.n_assets = 3;
    mc.n_days = 10;
    const Market m = simulate_market(mc);
    const std::string path = "/tmp/qubo_alloc_roundtrip.csv";
    save_returns_csv(m, path);
    const Market back = load_csv(path, false);
    CHECK(back.names == m.names);
    for (std::size_t t = 0; t < 10; ++t)
        for (std::size_t j = 0; j < 3; ++j) CHECK_NEAR(back.returns(t, j), m.returns(t, j), 1e-9);
    std::remove(path.c_str());
}

TEST(csv_errors_are_reported_with_line_numbers) {
    const std::string path = "/tmp/qubo_alloc_bad.csv";
    {
        std::ofstream f(path);
        f << "A,B\n100,50\n101,abc\n102,52\n";
    }
    bool ok = false;
    try {
        load_csv(path, true);
    } catch (const std::runtime_error& e) {
        ok = std::string(e.what()).find(":3:") != std::string::npos;
    }
    std::remove(path.c_str());
    CHECK(ok);
}

// -- laklaa.com
