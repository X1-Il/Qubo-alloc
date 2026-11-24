// Benchmark instances: realistic portfolio QUBOs drawn from the simulated
// market, so that solver benchmarks run on the structure the backtest
// actually produces (one dominant market mode, sector blocks, a penalty
// that dwarfs the risk/return terms) rather than on random couplings.
#pragma once

#include <cstdint>

#include "qubo/qubo.hpp"

namespace qubo {

// One-year window ending on a random day of a fresh simulated market.
PortfolioSpec random_portfolio_spec(int n_assets, int units, int bits, std::uint64_t seed,
                                    double risk_aversion = 4.0, double turnover = 0.05);

}  // namespace qubo

// -- laklaa.com
