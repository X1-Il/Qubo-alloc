cmake_minimum_required(VERSION 3.20)
project(qubo_alloc VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(QUBO_SANITIZE "Build with address and undefined-behaviour sanitizers" OFF)
option(QUBO_NATIVE "Tune for the build machine (-march=native)" OFF)

find_package(OpenMP REQUIRED)

add_library(qubo_core
  src/market.cpp
  src/estimators.cpp
  src/qubo.cpp
  src/instances.cpp
  src/backtest.cpp
  src/solvers/common.cpp
  src/solvers/exhaustive.cpp
  src/solvers/annealing.cpp
  src/solvers/sqa.cpp
)
target_include_directories(qubo_core PUBLIC include)
target_link_libraries(qubo_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(qubo_core PUBLIC -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Werror)
if(QUBO_NATIVE)
  target_compile_options(qubo_core PUBLIC -march=native)
endif()
if(QUBO_SANITIZE)
  target_compile_options(qubo_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(qubo_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(qubo-alloc src/main.cpp)
target_link_libraries(qubo-alloc PRIVATE qubo_core)

enable_testing()
add_executable(qubo_tests tests/test_main.cpp tests/test_core.cpp tests/test_solvers.cpp tests/test_backtest.cpp)
target_link_libraries(qubo_tests PRIVATE qubo_core)
add_test(NAME qubo_tests COMMAND qubo_tests)

# -- laklaa.com
