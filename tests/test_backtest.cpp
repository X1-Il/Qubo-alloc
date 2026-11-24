// Backtest tests: no look-ahead, accounting identities, baselines.

#include <algorithm>
#include <numeric>

#include "qubo/backtest.hpp"
#include "qubo/instances.hpp"
#include "qubo/market.hpp"
#include "test.hpp"

using namespace qubo;

TEST(simplex_projection_properties) {
    const std::vector<double> v{0.4, -1.0, 2.0, 0.3};
    const auto w = project_simplex(v);
    CHECK_NEAR(std::accumulate(w.begin(), w.end(), 0.0), 1.0, 1e-12);
    for (double x : w) CHECK(x >= 0.0);
    // Idempotent on points already in the simplex.
    const auto w2 = project_simplex(w);
    for (std::size_t i = 0; i < w.size(); ++i) CHECK_NEAR(w[i], w2[i], 1e-12);
    // Known answer: (2, 0.4, 0.3, -1) -> threshold 1, so only the 2 survives.
    CHECK_NEAR(w[2], 1.0, 1e-12);
}

TEST(relax_and_round_is_feasible) {
    for (int bits : {1, 2}) {
        const PortfolioSpec s = random_portfolio_spec(25, 8, bits, 50 + bits);
        const auto w = relax_and_round(s);
        double total = 0.0;
        for (double x : w) {
            total += x;
            const double u = x * s.budget_units;
            CHECK_NEAR(u, std::round(u), 1e-9);
            CHECK(u <= (1 << bits) - 1 + 1e-9);
        }
        CHECK_NEAR(total, 1.0, 1e-12);
    }
}

TEST(repair_lands_on_budget) {
    const PortfolioSpec s = random_portfolio_spec(12, 5, 1, 61);
    const PortfolioQubo pq = build_portfolio_qubo(s);
    for (int start : {0, 2, 9, 12}) {
        std::vector<int> u(12, 0);
        for (int a = 0; a < start; ++a) u[a] = 1;
        const auto w = repair_to_budget(pq, u);
        CHECK_NEAR(std::accumulate(w.begin(), w.end(), 0.0), 1.0, 1e-12);
    }
}

TEST(no_look_ahead_in_decisions) {
    // Scramble every return from day t onward. Decisions taken at or before t
    // must not move by a single bit.
    MarketConfig mc;
    mc.n_assets = 16;
    mc.n_days = 400;
    const Market base = simulate_market(mc);
    Market future_changed = base;
    const int t = 300;
    for (std::size_t d = t; d < future_changed.returns.rows(); ++d)
        for (std::size_t j = 0; j < 16; ++j) future_changed.returns(d, j) = -3.0 * future_changed.returns(d, j) + 0.01;
    BacktestConfig cfg;
    cfg.units = 5;
    cfg.sweeps = 100;
    for (Method m : {Method::RelaxRound, Method::SA, Method::SQA}) {
        const auto a = decide_weights(base, t, m, cfg, {});
        const auto b = decide_weights(future_changed, t, m, cfg, {});
        CHECK_MSG(a == b, method_name(m));
    }
}

TEST(backtest_accounting_is_consistent) {
    MarketConfig mc;
    mc.n_assets = 12;
    mc.n_days = 700;
    const Market m = simulate_market(mc);
    BacktestConfig cfg;
    cfg.units = 4;
    cfg.sweeps = 100;
    cfg.restarts = 2;
    const auto r = run_backtest(m, Method::SA, cfg);
    CHECK(r.daily.size() == static_cast<std::size_t>(mc.n_days - cfg.window));
    CHECK(r.equity.size() == r.daily.size() + 1);
    double eq = 1.0;
    for (double d : r.daily) eq *= 1.0 + d;
    CHECK_NEAR(eq, r.equity.back(), 1e-10);
    // First trade from cash buys exactly 100% of the book.
    CHECK_NEAR(r.turnover.front(), 1.0, 1e-12);
    // Holding exactly `units` names at every rebalance (bits = 1).
    CHECK_NEAR(r.metrics.avg_names, 4.0, 1e-12);
    CHECK(r.metrics.max_drawdown >= 0.0 && r.metrics.max_drawdown < 1.0);
}

TEST(transaction_costs_reduce_returns) {
    MarketConfig mc;
    mc.n_assets = 10;
    mc.n_days = 600;
    const Market m = simulate_market(mc);
    BacktestConfig cheap, dear;
    cheap.units = dear.units = 4;
    cheap.cost_bps = 0.0;
    dear.cost_bps = 50.0;
    const auto a = run_backtest(m, Method::RelaxRound, cheap);
    const auto b = run_backtest(m, Method::RelaxRound, dear);
    CHECK(b.equity.back() < a.equity.back());
}

// -- laklaa.com
