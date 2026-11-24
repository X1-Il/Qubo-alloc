#include "qubo/estimators.hpp"

#include <algorithm>
#include <stdexcept>

namespace qubo {

std::vector<double> column_means(const Matrix& X) {
    const std::size_t t_len = X.rows(), n = X.cols();
    std::vector<double> mu(n, 0.0);
    for (std::size_t t = 0; t < t_len; ++t) {
        const double* r = X.row(t);
        for (std::size_t j = 0; j < n; ++j) mu[j] += r[j];
    }
    for (auto& v : mu) v /= static_cast<double>(t_len);
    return mu;
}

Matrix sample_covariance(const Matrix& X) {
    const std::size_t t_len = X.rows(), n = X.cols();
    if (t_len < 2) throw std::invalid_argument("need at least two observations");
    const auto mu = column_means(X);
    Matrix s(n, n);
    std::vector<double> d(n);
    for (std::size_t t = 0; t < t_len; ++t) {
        const double* r = X.row(t);
        for (std::size_t j = 0; j < n; ++j) d[j] = r[j] - mu[j];
        // Upper triangle only; mirrored below. Halves the inner work.
        for (std::size_t i = 0; i < n; ++i) {
            double* si = s.row(i);
            const double di = d[i];
            for (std::size_t j = i; j < n; ++j) si[j] += di * d[j];
        }
    }
    const double inv = 1.0 / static_cast<double>(t_len);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            s(i, j) *= inv;
            s(j, i) = s(i, j);
        }
    }
    return s;
}

Shrinkage ledoit_wolf(const Matrix& X) {
    const std::size_t t_len = X.rows(), n = X.cols();
    const Matrix s = sample_covariance(X);
    const auto mu = column_means(X);
    const double nn = static_cast<double>(n);

    // The paper uses the normalised Frobenius norm ||A||^2 = tr(A A') / N.
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) m += s(i, i);
    m /= nn;

    double d2 = 0.0;  // ||S - m I||^2
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double v = s(i, j) - (i == j ? m : 0.0);
            d2 += v * v;
        }
    }
    d2 /= nn;

    // bbar^2 = (1/T^2) sum_t ||x_t x_t' - S||^2: how much a single
    // observation's outer product scatters around S. This is the
    // estimation-noise term, and the only one needing a pass over the data.
    double b2bar = 0.0;
    std::vector<double> d(n);
    for (std::size_t t = 0; t < t_len; ++t) {
        const double* r = X.row(t);
        for (std::size_t j = 0; j < n; ++j) d[j] = r[j] - mu[j];
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double v = d[i] * d[j] - s(i, j);
                acc += v * v;
            }
        }
        b2bar += acc / nn;
    }
    b2bar /= static_cast<double>(t_len) * static_cast<double>(t_len);

    const double b2 = std::min(b2bar, d2);
    const double intensity = d2 > 0.0 ? b2 / d2 : 1.0;

    Shrinkage out{Matrix(n, n), intensity};
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out.sigma(i, j) = (1.0 - intensity) * s(i, j) + (i == j ? intensity * m : 0.0);
        }
    }
    return out;
}

std::vector<double> james_stein_means(const Matrix& X) {
    const std::size_t t_len = X.rows(), n = X.cols();
    auto mu = column_means(X);
    if (n < 4) return mu;
    const Matrix s = sample_covariance(X);
    double grand = 0.0, avg_var = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        grand += mu[j];
        avg_var += s(j, j);
    }
    grand /= static_cast<double>(n);
    avg_var /= static_cast<double>(n);
    double spread = 0.0;
    for (double v : mu) spread += (v - grand) * (v - grand);
    // Positive-part James-Stein toward the grand mean. The noise variance of
    // each sample mean is avg_var / T.
    const double noise = avg_var / static_cast<double>(t_len);
    const double c = spread > 0.0 ? std::min(1.0, (static_cast<double>(n) - 3.0) * noise / spread) : 1.0;
    for (auto& v : mu) v = grand + (1.0 - c) * (v - grand);
    return mu;
}

}  // namespace qubo

// -- laklaa.com
