// Market data: a realistic simulated equity universe, or real returns
// from CSV.
//
// The simulator exists because a backtest needs thousands of days of
// returns with known ground truth, which lets us check that the
// estimators and the optimiser recover what was put in. It is not a toy:
//
//   r_{t,i} = alpha_i(t) / 252 + beta_i f^mkt_t + gamma_i f^sector(i)_t + eps_{t,i}
//
// - the market factor has GARCH(1,1) volatility (clustering, 16% a year
//   on average) and Student-t innovations (fat tails);
// - sector factors and idiosyncratic noise are Student-t as well;
// - alpha_i(t) is a slowly mean-reverting signal (half-life about six
//   months), so expected returns exist but are hard to estimate, as in
//   real markets.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qubo/matrix.hpp"

namespace qubo {

struct MarketConfig {
    int n_assets = 40;
    int n_days = 2520;  // ten years of trading days
    int n_sectors = 5;
    double market_vol = 0.16;      // annualised, long-run
    double sector_vol = 0.10;
    double idio_vol_min = 0.15;
    double idio_vol_max = 0.35;
    double mean_return = 0.06;     // annualised drift common to all assets
    double alpha_dispersion = 0.06;  // cross-sectional st. dev. of alpha, annualised
    double alpha_half_life = 126.0;  // days
    double tail_nu = 5.0;          // Student-t degrees of freedom
    double garch_alpha = 0.08;
    double garch_beta = 0.90;
    std::uint64_t seed = 7;
};

struct Market {
    std::vector<std::string> names;
    std::vector<int> sector;  // -1 when unknown (CSV input)
    Matrix returns;           // n_days x n_assets, daily log returns
};

Market simulate_market(const MarketConfig& cfg);

// CSV with a header row of asset names and one row per day. With
// `prices = true` the file holds prices and log returns are taken.
Market load_csv(const std::string& path, bool prices);
void save_returns_csv(const Market& m, const std::string& path);

}  // namespace qubo

// -- laklaa.com
