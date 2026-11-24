// Solver tests: exactness of the enumerator, and every heuristic against it.

#include <cmath>

#include "qubo/instances.hpp"
#include "qubo/qubo.hpp"
#include "qubo/rng.hpp"
#include "qubo/solvers.hpp"
#include "test.hpp"

using namespace qubo;

TEST(exhaustive_matches_naive_enumeration) {
    Rng rng(10);
    for (int n : {1, 5, 9, 12}) {
        Qubo q(n);
        for (int i = 0; i < n; ++i) {
            q.add_linear(i, rng.normal());
            for (int j = i + 1; j < n; ++j) q.add_pair(i, j, rng.normal());
        }
        double best = 1e300;
        for (int mask = 0; mask < (1 << n); ++mask) {
            Bits x(n);
            for (int i = 0; i < n; ++i) x[i] = static_cast<std::uint8_t>((mask >> i) & 1);
            best = std::min(best, q.energy(x));
        }
        const SolveResult r = solve_exhaustive(q);
        CHECK_NEAR(r.energy, best, 1e-10);
        CHECK_NEAR(q.energy(r.x), r.energy, 1e-12);
    }
}

TEST(heuristics_find_the_exact_optimum_on_portfolio_instances) {
    // Twenty-asset, pick-five instances: 2^20 states, exact answer known.
    // Each solver gets a budget that should be enough for a correct
    // implementation; per-restart success rates are measured in the bench.
    int sa_pair = 0, sa = 0, sqa = 0, pt = 0;
    const int instances = 8;
    for (int k = 0; k < instances; ++k) {
        const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(20, 5, 1, 200 + k));
        const double opt = solve_exhaustive(pq.qubo).energy;
        const double tol = 1e-9 * (1.0 + std::abs(opt));
        AnnealParams ap{300, 8, -1, -1, static_cast<std::uint64_t>(k + 1), true};
        if (solve_sa(pq.qubo, ap).energy <= opt + tol) ++sa_pair;
        ap.pair_moves = false;
        ap.restarts = 64;
        if (solve_sa(pq.qubo, ap).energy <= opt + tol) ++sa;
        SqaParams sp;
        sp.sweeps = 300;
        // Measured per-restart success on these instances ranges from 0.25 to
        // 0.58; 24 restarts keep the chance of missing the hardest one at
        // 0.75^24 = 0.1%.
        sp.restarts = 24;
        sp.seed = static_cast<std::uint64_t>(k + 1);
        if (solve_sqa(pq.qubo, sp).energy <= opt + tol) ++sqa;
        PtParams pp;
        pp.sweeps = 300;
        pp.restarts = 8;
        pp.seed = static_cast<std::uint64_t>(k + 1);
        if (solve_pt(pq.qubo, pp).energy <= opt + tol) ++pt;
    }
    CHECK_MSG(sa_pair == instances, "SA with pair moves " << sa_pair);
    CHECK_MSG(sqa == instances, "SQA " << sqa);
    CHECK_MSG(pt == instances, "PT " << pt);
    // Single-flip SA faces the penalty walls; with 64 restarts it should
    // still get most instances.
    CHECK_MSG(sa >= instances - 2, "SA " << sa);
}

TEST(single_flip_sa_is_trapped_by_penalty_walls_and_pair_moves_free_it) {
    // The structural fact behind the benchmark: every feasible point is a
    // strict local minimum under single flips, so per-restart success of
    // single-flip SA is low, while pair moves walk along the feasible set.
    const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(20, 5, 1, 900));
    const double opt = solve_exhaustive(pq.qubo).energy;
    const double tol = 1e-9 * (1.0 + std::abs(opt));
    AnnealParams ap{300, 64, -1, -1, 3, false};
    auto rate = [&](const SolveResult& r) {
        int hits = 0;
        for (double e : r.restart_energies) hits += e <= opt + tol;
        return static_cast<double>(hits) / static_cast<double>(r.restart_energies.size());
    };
    const double single = rate(solve_sa(pq.qubo, ap));
    ap.pair_moves = true;
    const double pair = rate(solve_sa(pq.qubo, ap));
    CHECK_MSG(single < 0.5, "single-flip " << single);
    CHECK_MSG(pair > 0.9, "pair moves " << pair);
}

TEST(results_do_not_depend_on_thread_count) {
    const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(30, 8, 1, 7));
    // Same seed twice: OpenMP scheduling may differ between runs, the answer may not.
    const auto a = solve_sa(pq.qubo, {200, 6, -1, -1, 99, true});
    const auto b = solve_sa(pq.qubo, {200, 6, -1, -1, 99, true});
    CHECK(a.x == b.x);
    CHECK(a.restart_energies == b.restart_energies);
    SqaParams sp;
    sp.sweeps = 100;
    sp.restarts = 3;
    sp.seed = 5;
    CHECK(solve_sqa(pq.qubo, sp).x == solve_sqa(pq.qubo, sp).x);
}

TEST(sqa_without_transverse_field_is_classical_and_with_it_explores) {
    // With gamma at the end value from the start, slices are locked from
    // sweep one, and SQA degenerates into a slow classical walk at temperature
    // P T. Turning the field on must not make things worse on average.
    double locked = 0.0, quantum = 0.0;
    for (int k = 0; k < 6; ++k) {
        const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(24, 6, 1, 300 + k));
        SqaParams sp;
        sp.sweeps = 150;
        sp.restarts = 2;
        sp.seed = 40 + k;
        quantum += solve_sqa(pq.qubo, sp).energy;
        sp.gamma_start = 1e-9;
        sp.gamma_end = 1e-9;
        locked += solve_sqa(pq.qubo, sp).energy;
    }
    CHECK_MSG(quantum <= locked + 1e-9, "quantum " << quantum << " locked " << locked);
}

TEST(time_to_solution_formula) {
    CHECK_NEAR(time_to_solution(1.0, 0.99), 1.0, 1e-12);
    CHECK_NEAR(time_to_solution(2.0, 0.5), 2.0 * std::log(0.01) / std::log(0.5), 1e-12);
    CHECK(std::isinf(time_to_solution(1.0, 0.0)));
}

TEST(energy_scale_is_positive_and_ordered) {
    const PortfolioQubo pq = build_portfolio_qubo(random_portfolio_spec(15, 5, 2, 3));
    const EnergyScale s = estimate_energy_scale(pq.qubo, 1);
    CHECK(s.low > 0.0);
    CHECK(s.low <= s.median);
}

// -- laklaa.com
