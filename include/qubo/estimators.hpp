// Estimators of expected returns and covariance from a window of returns.
//
// With N assets and T observations, the sample covariance has N(N+1)/2
// parameters estimated from N T numbers. For N = 40 and one year of data
// that is 820 parameters from about 10,000 noisy points, and the smallest
// eigenvalues come out far too small. A mean-variance optimiser then puts
// its weight exactly where the estimate is most wrong. Shrinkage is the
// standard cure, and the one used here is Ledoit and Wolf (2004).
#pragma once

#include <vector>

#include "qubo/matrix.hpp"

namespace qubo {

// Sample covariance with the 1/T normalisation (the MLE), which is the one
// the Ledoit-Wolf formulas are written for. X is T x N.
Matrix sample_covariance(const Matrix& X);

struct Shrinkage {
    Matrix sigma;
    double intensity = 0.0;  // weight on the target, in [0, 1]
};

// Ledoit and Wolf, "A well-conditioned estimator for large-dimensional
// covariance matrices", JMVA 88 (2004). Shrinks the sample covariance
// toward m I, m = average variance, with the intensity that minimises the
// expected Frobenius loss, estimated from the data alone.
Shrinkage ledoit_wolf(const Matrix& X);

// Column means of X.
std::vector<double> column_means(const Matrix& X);

// James-Stein shrinkage of the means toward their cross-sectional average.
// Sample means of daily returns are dominated by noise; pulling them toward
// the grand mean by the optimal positive-part factor reduces the total
// squared error whenever N >= 4.
std::vector<double> james_stein_means(const Matrix& X);

}  // namespace qubo

// -- laklaa.com
