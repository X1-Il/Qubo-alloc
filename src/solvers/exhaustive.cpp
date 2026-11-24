// Exact solver by Gray-code enumeration.
//
// Consecutive Gray codes differ in one bit, so walking all 2^n states costs
// one O(n) field update per state instead of an O(n^2) energy evaluation:
// 2^25 states in about a second. The space is split on the top bits into
// independent chunks for OpenMP; chunk results are reduced in a fixed
// order so ties break the same way whatever the thread count.

#include <bit>
#include <limits>
#include <stdexcept>

#include "internal.hpp"

namespace qubo {

SolveResult solve_exhaustive(const Qubo& q) {
    const int n = q.n;
    if (n < 1 || n > 32) throw std::invalid_argument("exhaustive search is limited to 1..32 variables");
    const double t0 = detail::now_seconds();

    // 2^6 = 64 chunks is plenty of parallel slack for any desktop CPU.
    const int top = std::min(n, 6);
    const int low = n - top;
    const long long chunks = 1LL << top;
    const unsigned long long per_chunk = 1ULL << low;

    std::vector<double> best_e(static_cast<std::size_t>(chunks), std::numeric_limits<double>::infinity());
    std::vector<Bits> best_x(static_cast<std::size_t>(chunks));

#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < chunks; ++c) {
        Bits x(n, 0);
        for (int b = 0; b < top; ++b) x[low + b] = static_cast<std::uint8_t>((c >> b) & 1);
        FieldState st(q, x);
        double be = st.energy();
        Bits bx = st.x();
        for (unsigned long long g = 1; g < per_chunk; ++g) {
            // Gray code step g flips the bit at the position of the lowest set bit of g.
            st.flip(std::countr_zero(g));
            if (st.energy() < be) {
                be = st.energy();
                bx = st.x();
            }
        }
        best_e[static_cast<std::size_t>(c)] = be;
        best_x[static_cast<std::size_t>(c)] = std::move(bx);
    }

    SolveResult res;
    res.restart_energies = best_e;
    res.sweeps = static_cast<long long>(chunks) * static_cast<long long>(per_chunk);
    detail::finish(q, res, best_x, t0);
    return res;
}

}  // namespace qubo

// -- laklaa.com
