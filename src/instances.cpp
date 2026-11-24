#include "qubo/instances.hpp"

#include "qubo/backtest.hpp"
#include "qubo/market.hpp"
#include "qubo/rng.hpp"

namespace qubo {

PortfolioSpec random_portfolio_spec(int n_assets, int units, int bits, std::uint64_t seed, double risk_aversion,
                                    double turnover) {
    MarketConfig mc;
    mc.n_assets = n_assets;
    mc.n_days = 600;
    mc.seed = seed;
    const Market m = simulate_market(mc);
    Rng rng(seed + 17);
    BacktestConfig cfg;
    cfg.units = units;
    cfg.bits = bits;
    cfg.risk_aversion = risk_aversion;
    cfg.turnover = turnover;
    const int t = cfg.window + static_cast<int>(rng.below(static_cast<std::uint64_t>(mc.n_days - cfg.window)));
    // A previous portfolio of `units` random names makes the turnover term bite.
    std::vector<double> w_prev(static_cast<std::size_t>(n_assets), 0.0);
    for (int k = 0; k < units; ++k) w_prev[rng.below(static_cast<std::uint64_t>(n_assets))] += 1.0 / units;
    return spec_at(m, t, cfg, w_prev);
}

}  // namespace qubo

// -- laklaa.com
