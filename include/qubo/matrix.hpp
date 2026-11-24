// A small dense row-major matrix.
//
// Portfolio problems here have at most a few hundred assets, so a plain
// contiguous buffer with explicit loops is both fast enough and easy to
// audit. No expression templates, no hidden allocations in hot loops.
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace qubo {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0) : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    static Matrix identity(std::size_t n) {
        Matrix m(n, n);
        for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    double& operator()(std::size_t i, std::size_t j) {
        assert(i < rows_ && j < cols_);
        return data_[i * cols_ + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        assert(i < rows_ && j < cols_);
        return data_[i * cols_ + j];
    }

    const double* row(std::size_t i) const { return data_.data() + i * cols_; }
    double* row(std::size_t i) { return data_.data() + i * cols_; }
    const std::vector<double>& data() const { return data_; }

    // y = A x
    std::vector<double> mul(const std::vector<double>& x) const {
        assert(x.size() == cols_);
        std::vector<double> y(rows_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i) {
            const double* r = row(i);
            double acc = 0.0;
            for (std::size_t j = 0; j < cols_; ++j) acc += r[j] * x[j];
            y[i] = acc;
        }
        return y;
    }

    // x' A x
    double quad(const std::vector<double>& x) const {
        const auto ax = mul(x);
        double acc = 0.0;
        for (std::size_t i = 0; i < rows_; ++i) acc += x[i] * ax[i];
        return acc;
    }

    Matrix& operator*=(double s) {
        for (auto& v : data_) v *= s;
        return *this;
    }

    double frobenius_distance(const Matrix& o) const {
        assert(rows_ == o.rows_ && cols_ == o.cols_);
        double acc = 0.0;
        for (std::size_t k = 0; k < data_.size(); ++k) acc += (data_[k] - o.data_[k]) * (data_[k] - o.data_[k]);
        return std::sqrt(acc);
    }

    // Largest eigenvalue of a symmetric PSD matrix by power iteration; used
    // for the Lipschitz constant of the relaxation, where 1% accuracy is
    // plenty and an eigen-solver would be overkill.
    double max_eigenvalue(int iters = 200) const {
        assert(rows_ == cols_);
        std::vector<double> v(rows_, 1.0 / std::sqrt(static_cast<double>(rows_)));
        double lambda = 0.0;
        for (int it = 0; it < iters; ++it) {
            auto w = mul(v);
            double norm = 0.0;
            for (double x : w) norm += x * x;
            norm = std::sqrt(norm);
            if (norm == 0.0) return 0.0;
            for (std::size_t i = 0; i < rows_; ++i) v[i] = w[i] / norm;
            lambda = norm;
        }
        return lambda;
    }

private:
    std::size_t rows_ = 0, cols_ = 0;
    std::vector<double> data_;
};

}  // namespace qubo

// -- laklaa.com
