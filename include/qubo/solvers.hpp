// QUBO solvers.
//
// - exhaustive: Gray-code enumeration, exact, n <= 32. The ground truth.
// - sa:   simulated annealing, Metropolis single flips, geometric cooling.
// - sqa:  simulated quantum annealing by path-integral Monte Carlo
//         (Suzuki-Trotter), the classical simulation of what a quantum
//         annealer does: a transverse field lets replicas "tunnel" through
//         barriers instead of climbing over them.
// - pt:   parallel tempering (replica exchange), the strongest general
//         classical heuristic for rugged landscapes and the honest
//         benchmark any quantum annealing claim should beat.
// - descent: random restarts of steepest descent, the floor.
//
// All MCMC solvers run independent restarts in parallel with OpenMP; each
// restart draws from its own stream Rng::stream(seed, r), so the answer
// does not depend on the thread count.
#pragma once

#include <cstdint>
#include <string>

#include "qubo/qubo.hpp"

namespace qubo {

struct SolveResult {
    Bits x;
    double energy = 0.0;
    double seconds = 0.0;
    long long sweeps = 0;  // total sweeps over all restarts and replicas
    // Best energy of each restart, for success-probability statistics.
    std::vector<double> restart_energies;
};

// Local fields f_i = h_i + sum_j J_ij x_j, kept in sync with x.
class FieldState {
public:
    FieldState(const Qubo& q, Bits x);
    double delta(int i) const { return (x_[i] ? -1.0 : 1.0) * field_[i]; }
    void flip(int i);  // O(n)
    double energy() const { return energy_; }
    const Bits& x() const { return x_; }
    const std::vector<double>& fields() const { return field_; }

private:
    const Qubo* q_;
    Bits x_;
    std::vector<double> field_;
    double energy_;
};

SolveResult solve_exhaustive(const Qubo& q);

struct AnnealParams {
    int sweeps = 1000;
    int restarts = 8;
    double t_start = -1.0;  // <= 0: calibrated from the energy landscape
    double t_end = -1.0;
    std::uint64_t seed = 1;
    // Also propose pair moves (one 1 -> 0 and one 0 -> 1 together, Kawasaki
    // dynamics). They keep the number of set bits, so with a cardinality
    // constraint they walk along the feasible set instead of climbing the
    // penalty wall. Off by default: hardware annealers only flip single
    // spins, and the benchmark keeps both versions to show the difference.
    bool pair_moves = false;
};
SolveResult solve_sa(const Qubo& q, const AnnealParams& p);

struct SqaParams {
    int sweeps = 1000;
    int restarts = 8;
    int slices = 32;            // Trotter slices P
    double temperature = -1.0;  // <= 0: calibrated
    double gamma_start = -1.0;  // transverse field at the start; <= 0: calibrated
    double gamma_end = -1.0;
    bool global_moves = true;   // also flip a spin in every slice at once
    std::uint64_t seed = 1;
};
SolveResult solve_sqa(const Qubo& q, const SqaParams& p);

struct PtParams {
    int sweeps = 1000;
    int restarts = 2;
    int replicas = 16;
    double t_min = -1.0;  // <= 0: calibrated
    double t_max = -1.0;
    std::uint64_t seed = 1;
};
SolveResult solve_pt(const Qubo& q, const PtParams& p);

SolveResult solve_descent(const Qubo& q, int restarts, std::uint64_t seed);

// Typical energy scales of the landscape, sampled around local minima:
// the wall height (median single-flip |dE|) and the fine scale (a low
// quantile of pair-move |dE|). Temperatures
// and fields are set from these so that no parameter needs hand-tuning per
// instance.
struct EnergyScale {
    double median = 1.0;  // typical single-flip |dE| out of a local minimum (wall height)
    double low = 1e-3;    // fine scale: small pair-move |dE| at local minima
};
EnergyScale estimate_energy_scale(const Qubo& q, std::uint64_t seed);

// Time to reach the target with probability `target` by independent
// repetitions: TTS = t ln(1 - target) / ln(1 - p). The standard metric of
// the annealing literature (Ronnow et al., Science 2014).
double time_to_solution(double seconds_per_run, double p_success, double target = 0.99);

}  // namespace qubo

// -- laklaa.com
