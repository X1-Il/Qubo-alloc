// Walk-forward backtest.
//
// At every rebalance date t, only returns strictly before t are used:
// a rolling window of `window` days feeds the Ledoit-Wolf covariance and
// the James-Stein means, the optimiser picks target weights, and the
// portfolio then drifts with the market until the next rebalance.
// Transaction costs are charged in basis points of traded notional.
//
// The QUBO methods all solve exactly the same problem at each date, so the
// backtest compares solvers as well as strategies: the penalty-free
// objective each method achieves is recorded date by date.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qubo/market.hpp"
#include "qubo/qubo.hpp"

namespace qubo {

enum class Method { EqualWeight, RelaxRound, Descent, SA, SAPair, SQA, PT, Exhaustive };
std::string method_name(Method m);
Method method_from_name(const std::string& s);

struct BacktestConfig {
    int window = 252;
    int rebalance = 21;
    int bits = 1;
    int units = 8;               // W; with bits = 1, the number of names held
    double risk_aversion = 4.0;
    double turnover = 0.05;      // tau in the objective
    double cost_bps = 10.0;      // charged on traded notional
    int sweeps = 400;
    int restarts = 8;
    std::uint64_t seed = 11;
};

struct Metrics {
    double cagr = 0.0;
    double ann_vol = 0.0;
    double sharpe = 0.0;  // excess over zero rate, annualised
    double max_drawdown = 0.0;
    double turnover_per_year = 0.0;
    double avg_names = 0.0;
};

struct BacktestResult {
    Method method;
    std::vector<double> daily;          // net simple returns
    std::vector<double> equity;         // starts at 1
    std::vector<int> rebalance_days;
    std::vector<double> objective;      // penalty-free QUBO objective at each rebalance
    std::vector<double> turnover;       // per rebalance, sum |dw|
    int infeasible_repaired = 0;
    double solver_seconds = 0.0;
    Metrics metrics;
};

// The problem the optimiser sees at day t, built from data before t only.
PortfolioSpec spec_at(const Market& m, int t, const BacktestConfig& cfg, const std::vector<double>& w_prev);

// Target weights of `method` at day t.
std::vector<double> decide_weights(const Market& m, int t, Method method, const BacktestConfig& cfg,
                                   const std::vector<double>& w_prev, double* solver_seconds = nullptr,
                                   bool* repaired = nullptr);

BacktestResult run_backtest(const Market& m, Method method, const BacktestConfig& cfg);

// Euclidean projection onto the probability simplex (Duchi et al., 2008).
std::vector<double> project_simplex(const std::vector<double>& v);

// Classical baseline: solve the continuous long-only problem by accelerated
// projected gradient, then round to the unit grid with largest remainders.
std::vector<double> relax_and_round(const PortfolioSpec& spec);

// Snap weights onto the feasible unit grid of `spec` (sum of units = W,
// each in [0, 2^bits - 1]) by greedy one-unit moves on the true objective.
// Used to repair the rare infeasible annealer output.
std::vector<double> repair_to_budget(const PortfolioQubo& pq, std::vector<int> units);

Metrics compute_metrics(const std::vector<double>& daily, const std::vector<double>& turnover,
                        const std::vector<double>& names_held);

}  // namespace qubo

// -- laklaa.com
