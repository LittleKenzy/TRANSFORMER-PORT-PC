#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tf::test {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        registry().push_back(TestCase{suite, name, std::move(fn)});
    }
};

#define TF_TEST(Suite, Name) \
    static void tf_test_##Suite##_##Name(); \
    static ::tf::test::Registrar tf_reg_##Suite##_##Name(#Suite, #Name, &tf_test_##Suite##_##Name); \
    static void tf_test_##Suite##_##Name()

#define TF_EXPECT(cond) \
    do { \
        if (!(cond)) { \
            std::ostringstream _oss; \
            _oss << __FILE__ << ":" << __LINE__ << " EXPECT failed: " #cond; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while (0)

#define TF_EXPECT_EQ(a, b) \
    do { \
        auto _va = (a); \
        auto _vb = (b); \
        if (!(_va == _vb)) { \
            std::ostringstream _oss; \
            _oss << __FILE__ << ":" << __LINE__ << " EXPECT_EQ failed: " #a " (" << _va << ") != " #b " (" << _vb << ")"; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while (0)

} // namespace tf::test
