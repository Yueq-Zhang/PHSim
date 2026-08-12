#include "common_function.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

template <typename Callable>
void expect_invalid_argument(const std::string& name, Callable&& callable) {
    try {
        callable();
        ++failures;
        std::cerr << "  FAIL: " << name
                  << " did not throw std::invalid_argument\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: " << name << " threw " << error.what() << '\n';
    } catch (...) {
        ++failures;
        std::cerr << "  FAIL: " << name << " threw a non-standard exception\n";
    }
}

}  // namespace

int main() {
    constexpr int invalid_value = -1;

    NPUStat npu_stat{};
    const auto invalid_npu = static_cast<NPUStat::StatType>(invalid_value);
    expect_invalid_argument("NPUStat::enum_to_string", [&] {
        (void)NPUStat::enum_to_string(invalid_npu);
    });
    expect_invalid_argument("NPUStat::get_by_enum", [&] {
        (void)npu_stat.get_by_enum(invalid_npu);
    });

    MemoryIOStat memory_stat{};
    const auto invalid_memory =
        static_cast<MemoryIOStat::StatType>(invalid_value);
    expect_invalid_argument("MemoryIOStat::enum_to_string", [&] {
        (void)MemoryIOStat::enum_to_string(invalid_memory);
    });
    expect_invalid_argument("MemoryIOStat::get_by_enum", [&] {
        (void)memory_stat.get_by_enum(invalid_memory);
    });

    OperationStat operation_stat("invalid-stat-test");
    const auto invalid_operation =
        static_cast<OperationStat::StatType>(invalid_value);
    expect_invalid_argument("OperationStat::enum_to_string", [&] {
        (void)OperationStat::enum_to_string(invalid_operation);
    });
    expect_invalid_argument("OperationStat::get_by_enum", [&] {
        (void)operation_stat.get_by_enum(invalid_operation);
    });

    if (failures == 0) {
        std::cout << "RESULT PASS: 6 invalid non-void paths throw in Release\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
