cmake_minimum_required(VERSION 3.16)
project(turboquant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library
add_library(turboquant
    src/pack.cpp
    src/quantize.cpp
)
target_include_directories(turboquant PUBLIC include)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(turboquant_tests
    tests/test_tile.cpp
    tests/test_pack.cpp
    tests/test_fp16.cpp
    tests/test_roundtrip.cpp
)
target_link_libraries(turboquant_tests turboquant GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(turboquant_tests)
