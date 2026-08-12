#include "common_function.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect_equal(const std::string& actual, const std::string& expected,
                  const std::string& name) {
    if (actual == expected) {
        std::cout << "  PASS: " << name << '\n';
        return;
    }

    ++failures;
    std::cerr << "  FAIL: " << name << " expected " << expected
              << ", got " << actual << '\n';
}

void test_known_memory_access_types() {
    const std::vector<std::pair<MemoryAccessType, std::string>> cases{
        {MemoryAccessType::READ, "READ"},
        {MemoryAccessType::WRITE, "WRITE"},
        {MemoryAccessType::GWRITE, "GWRITE"},
        {MemoryAccessType::COMP, "COMP"},
        {MemoryAccessType::COMP_HASH, "COMP_HASH"},
        {MemoryAccessType::READRES, "READRES"},
        {MemoryAccessType::P_HEADER, "P_HEADER"},
        {MemoryAccessType::COMPS_READRES, "COMPS_READRES"},
    };

    for (const auto& [type, expected] : cases) {
        expect_equal(memAccessTypeString(type), expected,
                     "MemoryAccessType " + expected);
    }
}

void test_invalid_memory_access_type() {
    try {
        (void)memAccessTypeString(static_cast<MemoryAccessType>(-1));
        ++failures;
        std::cerr << "  FAIL: invalid MemoryAccessType did not throw\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: invalid MemoryAccessType throws\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: invalid MemoryAccessType threw "
                  << error.what() << '\n';
    } catch (...) {
        ++failures;
        std::cerr << "  FAIL: invalid MemoryAccessType threw a non-standard exception\n";
    }
}

}  // namespace

int main() {
    test_known_memory_access_types();
    test_invalid_memory_access_type();

    if (failures == 0) {
        std::cout << "RESULT PASS: 9 runtime guard checks\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
