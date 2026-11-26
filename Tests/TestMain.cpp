#include "TestFramework.h"
#include <iostream>

int main() {
    int passed = 0;
    int failed = 0;

    for (const auto& tc : tf::test::registry()) {
        try {
            tc.fn();
            std::cout << "[PASS] " << tc.suite << "." << tc.name << "\n";
            passed++;
        } catch (const std::exception& ex) {
            std::cerr << "[FAIL] " << tc.suite << "." << tc.name << " -> " << ex.what() << "\n";
            failed++;
        }
    }

    std::cout << "\n=== TEST SUMMARY: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
