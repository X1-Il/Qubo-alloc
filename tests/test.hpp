// A deliberately small test harness: no network means no GoogleTest, and
// thirty lines are enough for registration, assertions with context, and
// a non-zero exit code on failure.
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace qtest {

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Failure {
    std::string what;
};

struct Register {
    Register(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

}  // namespace qtest

#define QT_CAT2(a, b) a##b
#define QT_CAT(a, b) QT_CAT2(a, b)
#define TEST(name)                                                       \
    static void name();                                                  \
    static qtest::Register QT_CAT(reg_, name)(#name, name);              \
    static void name()

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::ostringstream os_;                                                         \
            os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed";              \
            throw qtest::Failure{os_.str()};                                                \
        }                                                                                   \
    } while (0)

#define CHECK_MSG(cond, msg)                                                                \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::ostringstream os_;                                                         \
            os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed: " << msg;     \
            throw qtest::Failure{os_.str()};                                                \
        }                                                                                   \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                               \
    do {                                                                                    \
        const double a_ = (a), b_ = (b);                                                    \
        if (!(std::abs(a_ - b_) <= (tol))) {                                                \
            std::ostringstream os_;                                                         \
            os_.precision(17);                                                              \
            os_ << __FILE__ << ":" << __LINE__ << ": |" #a " - " #b "| = " << std::abs(a_ - b_) \
                << " > " << (tol) << " (" << a_ << " vs " << b_ << ")";                     \
            throw qtest::Failure{os_.str()};                                                \
        }                                                                                   \
    } while (0)

// -- laklaa.com
