// Random numbers: xoshiro256** seeded through SplitMix64.
//
// Every run of every solver is reproducible from one 64-bit seed, and
// parallel restarts derive their own streams from (seed, restart index),
// so results do not depend on how many threads OpenMP happens to use.
#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>

namespace qubo {

inline std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

class Rng {
public:
    explicit Rng(std::uint64_t seed) {
        std::uint64_t sm = seed;
        for (auto& w : s_) w = splitmix64(sm);
    }

    // A stream for sub-task `index`, independent of call order.
    static Rng stream(std::uint64_t seed, std::uint64_t index) {
        std::uint64_t sm = seed ^ (0xD1B54A32D192ED03ULL * (index + 1));
        return Rng(splitmix64(sm));
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform on [0, 1) with 53 bits.
    double uniform() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }

    // Unbiased integer in [0, n). Draws below 2^64 mod n are rejected so
    // that every residue has exactly the same number of preimages; the
    // rejection probability is below n / 2^64, i.e. never in practice.
    std::uint64_t below(std::uint64_t n) {
        const std::uint64_t threshold = (0 - n) % n;
        for (;;) {
            const std::uint64_t r = next_u64();
            if (r >= threshold) return r % n;
        }
    }

    // Standard normal, Box-Muller with the spare cached.
    double normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        const double u1 = 1.0 - uniform();  // (0, 1], log is finite
        const double u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = 2.0 * std::numbers::pi * u2;
        spare_ = r * std::sin(th);
        has_spare_ = true;
        return r * std::cos(th);
    }

    // Gamma(shape, 1), Marsaglia and Tsang (2000).
    double gamma(double shape) {
        if (shape < 1.0) {
            // Boost to shape + 1 and correct with a uniform power.
            return gamma(shape + 1.0) * std::pow(1.0 - uniform(), 1.0 / shape);
        }
        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x, v;
            do {
                x = normal();
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            const double u = 1.0 - uniform();
            if (std::log(u) < 0.5 * x * x + d - d * v + d * std::log(v)) return d * v;
        }
    }

    // Student t with `nu` degrees of freedom, rescaled to unit variance so
    // that it can replace a normal innovation without changing volatility.
    double student_t_unit(double nu) {
        const double chi2 = 2.0 * gamma(0.5 * nu);
        const double t = normal() / std::sqrt(chi2 / nu);
        return t * std::sqrt((nu - 2.0) / nu);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    std::uint64_t s_[4]{};
    double spare_ = 0.0;
    bool has_spare_ = false;
};

}  // namespace qubo

// -- laklaa.com
