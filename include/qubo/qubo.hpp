// QUBO model and the portfolio formulation.
//
// A QUBO is E(x) = offset + sum_i h_i x_i + sum_{i<j} J_ij x_i x_j over
// x in {0,1}^n. It is the native input of quantum annealers (D-Wave) and
// of QAOA-style gate-model algorithms, and any binary quadratic problem
// with linear equality constraints maps to it through quadratic penalties.
//
// J is stored as a full symmetric matrix with a zero diagonal, which makes
// the local field f_i = h_i + sum_j J_ij x_j a single contiguous row scan.
// Flipping x_i changes the energy by (1 - 2 x_i) f_i, so every MCMC
// solver can evaluate a move in O(1) and update the fields in O(n) when
// the move is accepted. That incremental update is what makes the solvers
// fast; recomputing E from scratch would be O(n^2) per proposal.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qubo/matrix.hpp"

namespace qubo {

using Bits = std::vector<std::uint8_t>;

struct Qubo {
    int n = 0;
    std::vector<double> h;  // linear terms
    std::vector<double> J;  // n*n, symmetric, zero diagonal: coefficient of x_i x_j for i != j
    double offset = 0.0;

    explicit Qubo(int n_vars = 0) : n(n_vars), h(n_vars, 0.0), J(static_cast<std::size_t>(n_vars) * n_vars, 0.0) {}

    double coupling(int i, int j) const { return J[static_cast<std::size_t>(i) * n + j]; }
    // Adds c to the coefficient of x_i x_j. i == j folds into h, since x^2 = x.
    void add_pair(int i, int j, double c);
    void add_linear(int i, double c) { h[i] += c; }

    // Full O(n^2) evaluation, used as the reference in tests.
    double energy(const Bits& x) const;
};

// Ising form: E(s) = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j, s in {-1, +1},
// obtained by substituting x = (1 + s) / 2. Simulated quantum annealing works
// in spins because the transverse field and the Trotter coupling are
// naturally written in them.
struct Ising {
    int n = 0;
    std::vector<double> h, J;
    double offset = 0.0;
    double energy(const std::vector<int>& s) const;
};
Ising to_ising(const Qubo& q);

// ---------------------------------------------------------------------
// Portfolio construction as a QUBO
// ---------------------------------------------------------------------
//
// Each asset gets `bits` binary variables encoding an integer number of
// "units" u_i in [0, 2^bits - 1]; the weight is w_i = u_i / W for a budget
// of W units. bits = 1 with W = K is the classic cardinality-constrained
// problem: pick exactly K assets, equally weighted.
//
//   minimise   gamma w' Sigma w  -  mu' w  +  tau ||w - w_prev||^2  +  A (sum_i u_i - W)^2
//
// The turnover term is quadratic (a market-impact style cost) rather than
// an absolute value. For bits = 1 it is exactly linear in x, because
// x^2 = x, and in general it keeps the problem a QUBO without auxiliary
// variables.

struct PortfolioSpec {
    Matrix sigma;               // annualised covariance
    std::vector<double> mu;     // annualised expected returns
    std::vector<double> w_prev; // current weights (empty = all cash)
    double risk_aversion = 4.0; // gamma
    double turnover = 0.0;      // tau
    int bits = 1;
    int budget_units = 8;       // W
    double penalty = -1.0;      // A; negative means calibrate automatically
};

struct PortfolioQubo {
    Qubo qubo;
    PortfolioSpec spec;
    int n_assets = 0;
    double penalty = 0.0;

    std::vector<int> units(const Bits& x) const;
    std::vector<double> weights(const Bits& x) const;
    bool feasible(const Bits& x) const;
    // The financial objective without the penalty, from weights directly.
    // Tests use it to check that the QUBO coefficients are right.
    double objective(const std::vector<double>& w) const;
};

PortfolioQubo build_portfolio_qubo(const PortfolioSpec& spec);

}  // namespace qubo

// -- laklaa.com
