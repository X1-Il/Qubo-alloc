// Helpers shared by the solver implementations. Not part of the public API.
#pragma once

#include <chrono>
#include <cmath>
#include <vector>

#include "qubo/rng.hpp"
#include "qubo/solvers.hpp"

namespace qubo::detail {

inline double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline Bits random_bits(int n, Rng& rng) {
    Bits x(n);
    for (auto& b : x) b = static_cast<std::uint8_t>(rng.below(2));
    return x;
}

// One Metropolis sweep at inverse temperature beta, in index order.
// Tracks the best state seen. Sequential sweeps mix as well as random-site
// updates for these problems and are friendlier to the cache and the
// branch predictor.
inline void metropolis_sweep(FieldState& st, double beta, Rng& rng, double& best_e, Bits& best_x) {
    const int n = static_cast<int>(st.x().size());
    for (int i = 0; i < n; ++i) {
        const double d = st.delta(i);
        if (d <= 0.0 || rng.uniform() < std::exp(-beta * d)) {
            st.flip(i);
            if (st.energy() < best_e) {
                best_e = st.energy();
                best_x = st.x();
            }
        }
    }
}

// Picks the best restart and fills in the summary fields.
inline void finish(const Qubo& q, SolveResult& res, std::vector<Bits>& per_restart_x, double t0) {
    std::size_t best = 0;
    for (std::size_t r = 1; r < res.restart_energies.size(); ++r) {
        if (res.restart_energies[r] < res.restart_energies[best]) best = r;
    }
    res.x = std::move(per_restart_x[best]);
    // Re-evaluate from scratch: the incrementally tracked energy drifts by
    // rounding over millions of updates; reported numbers should not.
    res.energy = q.energy(res.x);
    res.seconds = now_seconds() - t0;
}

}  // namespace qubo::detail

// -- laklaa.com
