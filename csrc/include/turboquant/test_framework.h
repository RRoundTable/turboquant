#pragma once

// Minimal test framework (no dependencies)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace tq_test {

struct TestCase {
    std::string name;
    std::function<void()> func;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& fail_count() {
    static int count = 0;
    return count;
}

inline const char*& current_test() {
    static const char* name = nullptr;
    return name;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline void check_impl(bool cond, const char* expr, const char* file, int line,
                        const char* msg = nullptr) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s:%d: %s", file, line, expr);
        if (msg) std::fprintf(stderr, " -- %s", msg);
        std::fprintf(stderr, "\n");
        fail_count()++;
    }
}

inline int run_all() {
    int passed = 0, failed = 0;
    for (auto& tc : registry()) {
        current_test() = tc.name.c_str();
        int before = fail_count();
        tc.func();
        if (fail_count() == before) {
            std::printf("  PASS: %s\n", tc.name.c_str());
            passed++;
        } else {
            std::printf("  FAIL: %s\n", tc.name.c_str());
            failed++;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}

}  // namespace tq_test

#define TEST(suite, name)                                                     \
    static void suite##_##name##_body();                                      \
    static tq_test::Registrar suite##_##name##_reg(                           \
        #suite "." #name, suite##_##name##_body);                             \
    static void suite##_##name##_body()

#define EXPECT_EQ(a, b)                                                       \
    tq_test::check_impl((a) == (b), #a " == " #b, __FILE__, __LINE__)

#define EXPECT_LT(a, b)                                                       \
    tq_test::check_impl((a) < (b), #a " < " #b, __FILE__, __LINE__)

#define EXPECT_LE(a, b)                                                       \
    tq_test::check_impl((a) <= (b), #a " <= " #b, __FILE__, __LINE__)

#define EXPECT_GE(a, b)                                                       \
    tq_test::check_impl((a) >= (b), #a " >= " #b, __FILE__, __LINE__)

#define EXPECT_GT(a, b)                                                       \
    tq_test::check_impl((a) > (b), #a " > " #b, __FILE__, __LINE__)

#define EXPECT_TRUE(x)                                                        \
    tq_test::check_impl(!!(x), #x, __FILE__, __LINE__)

#define EXPECT_FLOAT_EQ(a, b)                                                 \
    tq_test::check_impl(std::fabs((a) - (b)) < 1e-6f,                        \
                         #a " ~= " #b, __FILE__, __LINE__)

#define RUN_ALL_TESTS() tq_test::run_all()
