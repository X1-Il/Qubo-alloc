// Simulated annealing and parallel tempering.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "internal.hpp"

namespace qubo {

namespace {

// Start hot enough that a typical uphill move is accepted half the time,
// end cold enough that even a small uphill move is accepted 1% of the time
// at most. Both read off the sampled distribution of |dE|.
std::pair<double, double> calibrated_range(const Qubo& q, double t_hi, double t_lo, std::uint64_t seed) {
    if (t_hi > 0.0 && t_lo > 0.0) return {t_hi, t_lo};
    const EnergyScale s = estimate_energy_scale(q, seed);
    const double hi = t_hi > 0.0 ? t_hi : s.median / std::log(2.0);
    // At the end, an uphill pair move of the fine scale is accepted 1% of
    // the time: cold enough to settle into the best nearby solution.
    const double lo = t_lo > 0.0 ? t_lo : s.low / std::log(100.0);
    return {hi, std::min(lo, hi)};
}

}  // namespace

SolveResult solve_sa(const Qubo& q, const AnnealParams& p) {
    if (p.sweeps < 1 || p.restarts < 1) throw std::invalid_argument("sweeps and restarts must be positive");
    const double t0 = detail::now_seconds();
    const auto [t_start, t_end] = calibrated_range(q, p.t_start, p.t_end, p.seed);
    // Geometric cooling: equal time per decade of temperature, which suits
    // landscapes whose barriers span several orders of magnitude (here the
    // penalty barrier and the much smaller risk/return differences).
    const double ratio = p.sweeps > 1 ? std::pow(t_end / t_start, 1.0 / (p.sweeps - 1)) : 1.0;

    SolveResult res;
    res.restart_energies.assign(p.restarts, 0.0);
    std::vector<Bits> best_x(p.restarts);

#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < p.restarts; ++r) {
        Rng rng = Rng::stream(p.seed, static_cast<std::uint64_t>(r));
        FieldState st(q, detail::random_bits(q.n, rng));
        double be = st.energy();
        Bits bx = st.x();
        double t = t_start;
        std::vector<int> ones, zeros;
        for (int s = 0; s < p.sweeps; ++s, t *= ratio) {
            const double beta = 1.0 / t;
            detail::metropolis_sweep(st, beta, rng, be, bx);
            if (!p.pair_moves) continue;
            // n pair proposals per sweep, each a uniformly random (1, 0) pair.
            ones.clear();
            zeros.clear();
            for (int i = 0; i < q.n; ++i) (st.x()[i] ? ones : zeros).push_back(i);
            if (ones.empty() || zeros.empty()) continue;
            for (int m = 0; m < q.n; ++m) {
                const auto oi = rng.below(ones.size()), zi = rng.below(zeros.size());
                const int i = ones[oi], j = zeros[zi];
                const double d = st.delta(i) + st.delta(j) - q.coupling(i, j);
                if (d <= 0.0 || rng.uniform() < std::exp(-beta * d)) {
                    st.flip(i);
                    st.flip(j);
                    ones[oi] = j;
                    zeros[zi] = i;
                    if (st.energy() < be) {
                        be = st.energy();
                        bx = st.x();
                    }
                }
            }
        }
        res.restart_energies[r] = be;
        best_x[r] = std::move(bx);
    }
    res.sweeps = static_cast<long long>(p.sweeps) * p.restarts;
    detail::finish(q, res, best_x, t0);
    return res;
}

SolveResult solve_pt(const Qubo& q, const PtParams& p) {
    if (p.sweeps < 1 || p.restarts < 1 || p.replicas < 2) throw std::invalid_argument("invalid parallel tempering parameters");
    const double t0 = detail::now_seconds();
    // The hottest replica sits three times above the SA starting point: it
    // must cross walls freely and feed the ladder with fresh configurations
    // (measured: 30% per-run success at 3x, 20% at 1x on 20-asset instances).
    auto [t_max, t_min] = calibrated_range(q, p.t_max, p.t_min, p.seed);
    if (p.t_max <= 0.0) t_max *= 3.0;
    const int R = p.replicas;
    // Geometric ladder: constant ratio between neighbours gives roughly
    // uniform swap acceptance when the heat capacity is flat.
    std::vector<double> beta(R);
    for (int k = 0; k < R; ++k) {
        const double t = t_min * std::pow(t_max / t_min, static_cast<double>(k) / (R - 1));
        beta[k] = 1.0 / t;
    }

    SolveResult res;
    res.restart_energies.assign(p.restarts, 0.0);
    std::vector<Bits> best_x(p.restarts);

#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < p.restarts; ++r) {
        Rng rng = Rng::stream(p.seed, static_cast<std::uint64_t>(r));
        std::vector<FieldState> reps;
        reps.reserve(R);
        for (int k = 0; k < R; ++k) reps.emplace_back(q, detail::random_bits(q.n, rng));
        // slot[k] = index of the replica currently at temperature k. Swapping
        // two integers is far cheaper than swapping configurations.
        std::vector<int> slot(R);
        std::iota(slot.begin(), slot.end(), 0);
        double be = std::numeric_limits<double>::infinity();
        Bits bx;
        for (const auto& rep : reps) {
            if (rep.energy() < be) {
                be = rep.energy();
                bx = rep.x();
            }
        }
        for (int s = 0; s < p.sweeps; ++s) {
            for (int k = 0; k < R; ++k) detail::metropolis_sweep(reps[slot[k]], beta[k], rng, be, bx);
            // Alternate even and odd neighbour pairs so that every pair gets
            // a chance and information can travel the whole ladder.
            for (int k = s % 2; k + 1 < R; k += 2) {
                const double ea = reps[slot[k]].energy(), eb = reps[slot[k + 1]].energy();
                const double log_acc = (beta[k] - beta[k + 1]) * (ea - eb);
                if (log_acc >= 0.0 || rng.uniform() < std::exp(log_acc)) std::swap(slot[k], slot[k + 1]);
            }
        }
        res.restart_energies[r] = be;
        best_x[r] = std::move(bx);
    }
    res.sweeps = static_cast<long long>(p.sweeps) * p.restarts * R;
    detail::finish(q, res, best_x, t0);
    return res;
}

}  // namespace qubo

// -- laklaa.com
