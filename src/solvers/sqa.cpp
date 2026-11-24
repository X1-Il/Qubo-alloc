// Simulated quantum annealing by path-integral Monte Carlo.
//
// A quantum annealer evolves H(t) = H_problem + Gamma(t) H_driver, with a
// transverse-field driver -sum_i sigma^x_i whose strength Gamma goes from
// large to zero. Through the Suzuki-Trotter decomposition, the thermal
// state of that Hamiltonian at temperature T maps to P classical copies
// ("Trotter slices") of the spin system, coupled along the imaginary-time
// direction (Martonak, Santoro, Tosatti, PRB 66, 094203, 2002):
//
//   E_eff = sum_k H_problem(s^k) / P  -  J_perp sum_k sum_i s_i^k s_i^{k+1},
//   J_perp = -(P T / 2) ln tanh(Gamma / (P T))   (periodic in k)
//
// Large Gamma: J_perp ~ 0, the slices are independent and explore freely.
// Gamma -> 0: J_perp grows and the slices are forced to agree, collapsing
// onto one classical configuration. Between the two, a slice can sit on the
// other side of a barrier while its neighbours have not crossed yet, which
// is the classical shadow of tunnelling. Metropolis runs at temperature T
// on E_eff.
//
// SQA is a classical algorithm. On some landscapes it tracks a quantum
// annealer's behaviour closely, which is why it is the standard way to ask
// "would quantum annealing help here?" without hardware.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "internal.hpp"

namespace qubo {

SolveResult solve_sqa(const Qubo& q, const SqaParams& p) {
    if (p.sweeps < 1 || p.restarts < 1 || p.slices < 2) throw std::invalid_argument("invalid SQA parameters");
    const double t0 = detail::now_seconds();
    const int P = p.slices;
    const int n = q.n;

    // Calibration. P T is the temperature each slice's copy of the problem
    // sees. It has to be cold relative to the constraint walls (otherwise
    // the final state is a random feasible portfolio) but not frozen
    // (otherwise the Trotter dynamics stop). A grid search on 10 instances
    // with known optima gave a plateau around P T = 0.3 x wall height with a
    // cliff to zero success at 0.1, and a starting field of 1 to 3 x wall
    // height (slices free to disagree across a wall). The field ends two
    // decades below the fine objective scale, where the slices are locked.
    const EnergyScale scale = estimate_energy_scale(q, p.seed);
    const double PT = p.temperature > 0.0 ? p.temperature * P : 0.3 * scale.median;
    const double T = PT / P;
    const double g_start = p.gamma_start > 0.0 ? p.gamma_start : 3.0 * scale.median;
    const double g_end = p.gamma_end > 0.0 ? p.gamma_end : 0.01 * scale.low;

    SolveResult res;
    res.restart_energies.assign(p.restarts, 0.0);
    std::vector<Bits> best_x(p.restarts);

#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < p.restarts; ++r) {
        Rng rng = Rng::stream(p.seed, static_cast<std::uint64_t>(r));
        std::vector<FieldState> slice;
        slice.reserve(P);
        // All slices start from the same random state: the classical limit
        // of the uniform superposition is not a single state, but starting
        // aligned avoids paying a huge coupling energy at the first sweep.
        const Bits start = detail::random_bits(n, rng);
        for (int k = 0; k < P; ++k) slice.emplace_back(q, start);

        double be = slice[0].energy();
        Bits bx = slice[0].x();
        const double beta = 1.0 / T;
        for (int s = 0; s < p.sweeps; ++s) {
            // Linear decrease of Gamma, the schedule used on D-Wave hardware.
            const double frac = p.sweeps > 1 ? static_cast<double>(s) / (p.sweeps - 1) : 1.0;
            const double gamma = g_start + (g_end - g_start) * frac;
            const double j_perp = -0.5 * PT * std::log(std::tanh(gamma / PT));

            // Local moves: one spin in one slice.
            for (int k = 0; k < P; ++k) {
                const FieldState& up = slice[(k + 1) % P];
                const FieldState& down = slice[(k + P - 1) % P];
                FieldState& cur = slice[k];
                for (int i = 0; i < n; ++i) {
                    const double si = cur.x()[i] ? 1.0 : -1.0;
                    const double neigh = (up.x()[i] ? 1.0 : -1.0) + (down.x()[i] ? 1.0 : -1.0);
                    // Flipping s_i changes -J_perp s_i (s_up + s_down) by +2 J_perp s_i (s_up + s_down).
                    const double d = cur.delta(i) / P + 2.0 * j_perp * si * neigh;
                    if (d <= 0.0 || rng.uniform() < std::exp(-beta * d)) cur.flip(i);
                }
            }
            // Global moves: the same spin in every slice. The Trotter term is
            // unchanged, so these are pure problem-energy moves that let the
            // whole path cross a barrier together once the slices are locked.
            if (p.global_moves) {
                for (int i = 0; i < n; ++i) {
                    double d = 0.0;
                    for (int k = 0; k < P; ++k) d += slice[k].delta(i);
                    d /= P;
                    if (d <= 0.0 || rng.uniform() < std::exp(-beta * d)) {
                        for (int k = 0; k < P; ++k) slice[k].flip(i);
                    }
                }
            }
            for (const auto& sl : slice) {
                if (sl.energy() < be) {
                    be = sl.energy();
                    bx = sl.x();
                }
            }
        }
        res.restart_energies[r] = be;
        best_x[r] = std::move(bx);
    }
    res.sweeps = static_cast<long long>(p.sweeps) * p.restarts * P;
    detail::finish(q, res, best_x, t0);
    return res;
}

}  // namespace qubo

// -- laklaa.com
