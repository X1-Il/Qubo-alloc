#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "internal.hpp"

namespace qubo {

FieldState::FieldState(const Qubo& q, Bits x) : q_(&q), x_(std::move(x)), field_(q.h), energy_(q.offset) {
    if (static_cast<int>(x_.size()) != q.n) throw std::invalid_argument("state size differs from QUBO size");
    const int n = q.n;
    for (int i = 0; i < n; ++i) {
        if (!x_[i]) continue;
        energy_ += q.h[i];
        const double* row = q.J.data() + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) field_[j] += row[j];
    }
    // Pair terms were not yet in the energy: sum_{i<j} J_ij x_i x_j
    // = 1/2 sum_i x_i (f_i - h_i).
    double pair = 0.0;
    for (int i = 0; i < n; ++i) {
        if (x_[i]) pair += field_[i] - q.h[i];
    }
    energy_ += 0.5 * pair;
}

void FieldState::flip(int i) {
    const int n = q_->n;
    energy_ += delta(i);
    const double s = x_[i] ? -1.0 : 1.0;
    x_[i] ^= 1;
    // J is symmetric, so row i is also column i: contiguous and vectorisable.
    const double* row = q_->J.data() + static_cast<std::size_t>(i) * n;
    double* f = field_.data();
    for (int j = 0; j < n; ++j) f[j] += s * row[j];
}

EnergyScale estimate_energy_scale(const Qubo& q, std::uint64_t seed) {
    Rng rng(seed ^ 0x5DEECE66DULL);
    // Coarse scale: single-flip moves out of local minima, i.e. the height
    // of the walls a solver must climb to move between solutions. Sampling
    // at random states instead would measure how violated a random point is,
    // which grows with n and would waste most of an anneal at temperatures
    // far above anything relevant.
    std::vector<double> coarse;
    // Fine scale: pair moves (one bit up, one bit down) at local minima.
    // These connect neighbouring feasible solutions without leaving the
    // constraint surface, so they measure the objective differences the
    // solver must finally resolve, which can be hundreds of times smaller.
    std::vector<double> fine;
    for (int sample = 0; sample < 8; ++sample) {
        FieldState st(q, detail::random_bits(q.n, rng));
        for (;;) {  // steepest descent to a local minimum
            int bi = -1;
            double bd = -1e-12;
            for (int i = 0; i < q.n; ++i) {
                if (st.delta(i) < bd) {
                    bd = st.delta(i);
                    bi = i;
                }
            }
            if (bi < 0) break;
            st.flip(bi);
        }
        for (int i = 0; i < q.n; ++i) {
            const double d = std::abs(st.delta(i));
            if (d > 1e-12) coarse.push_back(d);
        }
        for (int i = 0; i < q.n; ++i) {
            if (!st.x()[i]) continue;
            for (int j = 0; j < q.n; ++j) {
                if (st.x()[j]) continue;
                // Flip i (1 -> 0) then j (0 -> 1): the second delta sees the
                // first through the coupling, (1 - 2 x_i)(1 - 2 x_j) J_ij = -J_ij.
                const double d = std::abs(st.delta(i) + st.delta(j) - q.coupling(i, j));
                if (d > 1e-12) fine.push_back(d);
            }
        }
    }
    EnergyScale s;
    if (!coarse.empty()) {
        std::sort(coarse.begin(), coarse.end());
        s.median = coarse[coarse.size() / 2];
        s.low = coarse[coarse.size() / 20];
    }
    if (!fine.empty()) {
        std::sort(fine.begin(), fine.end());
        s.low = std::min(s.low, fine[fine.size() / 10]);
    }
    return s;
}

double time_to_solution(double seconds_per_run, double p_success, double target) {
    if (p_success <= 0.0) return std::numeric_limits<double>::infinity();
    if (p_success >= target) return seconds_per_run;
    return seconds_per_run * std::log(1.0 - target) / std::log(1.0 - p_success);
}

SolveResult solve_descent(const Qubo& q, int restarts, std::uint64_t seed) {
    SolveResult res;
    res.restart_energies.assign(restarts, 0.0);
    std::vector<Bits> best_x(restarts);
    const auto start = detail::now_seconds();
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < restarts; ++r) {
        Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(r));
        FieldState st(q, detail::random_bits(q.n, rng));
        for (;;) {
            int best_i = -1;
            double best_d = -1e-12;
            for (int i = 0; i < q.n; ++i) {
                const double d = st.delta(i);
                if (d < best_d) {
                    best_d = d;
                    best_i = i;
                }
            }
            if (best_i < 0) break;
            st.flip(best_i);
        }
        res.restart_energies[r] = st.energy();
        best_x[r] = st.x();
    }
    detail::finish(q, res, best_x, start);
    return res;
}

}  // namespace qubo

// -- laklaa.com
