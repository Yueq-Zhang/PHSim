#include "Client/RequestGenerator.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Expected valid and invalid request-trace paths\n";
        return 1;
    }

    int failures = 0;
    const std::string valid_path = argv[1];
    const std::string invalid_path = argv[2];

    try {
        RequestGenerator::init(valid_path, 1);
        const auto first = RequestGenerator::get_qa_length();
        if (RequestGenerator::get_total_req_cnt() != 1 || first.first != 4 ||
            first.second != 7 || RequestGenerator::has_data()) {
            ++failures;
            std::cerr << "Valid trace did not produce the expected 4/7 row\n";
        }

        // A second initialization in the same process must replace, not append
        // to, the process-global parser state.
        RequestGenerator::init(valid_path, 1);
        if (RequestGenerator::get_total_req_cnt() != 1) {
            ++failures;
            std::cerr << "Repeated trace initialization accumulated old rows\n";
        }
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "Valid trace was rejected: " << error.what() << '\n';
    }

    try {
        RequestGenerator::init(invalid_path, 1);
        ++failures;
        std::cerr << "Invalid numeric cell was accepted\n";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find(":2, column 2") == std::string::npos) {
            ++failures;
            std::cerr << "Invalid-cell diagnostic omitted location: "
                      << message << '\n';
        }
    }

    try {
        RequestGenerator::init(valid_path + ".missing", 1);
        ++failures;
        std::cerr << "Missing trace file was accepted\n";
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("Cannot open request trace file") ==
            std::string::npos) {
            ++failures;
            std::cerr << "Missing-file diagnostic was unclear: "
                      << error.what() << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << "RESULT FAIL: " << failures
                  << " RequestGenerator check(s) failed\n";
        return 1;
    }
    std::cout << "RESULT PASS: RequestGenerator CSV validation\n";
    return 0;
}
