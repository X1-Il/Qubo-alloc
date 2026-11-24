#include <chrono>
#include <cstring>

#include "test.hpp"

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int passed = 0, failed = 0;
    for (const auto& c : qtest::registry()) {
        if (filter && !std::strstr(c.name.c_str(), filter)) continue;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            c.fn();
            ++passed;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "[ ok ] " << c.name << " (" << static_cast<int>(ms) << " ms)\n";
        } catch (const qtest::Failure& f) {
            ++failed;
            std::cout << "[FAIL] " << c.name << "\n       " << f.what << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[FAIL] " << c.name << "\n       exception: " << e.what() << "\n";
        }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}

// -- laklaa.com
