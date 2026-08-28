#include "RequestGenerator.h"

namespace RequestGenerator {
uint32_t answer_index;
uint32_t row_index;
std::vector<std::string> columns;
std::vector<std::vector<uint32_t>> table;

void init(std::string path, uint32_t _answer_index) {
    row_index = 0;
    columns.clear();
    table.clear();

    // todo
    // initialize answer_index depending on the file type
    answer_index = _answer_index;

    parse(path);
    spdlog::info("parsed {} lines from file {}", table.size(), path);
}
int get_total_req_cnt() { return table.size(); }

bool has_data() { return row_index < table.size(); }

std::pair<uint32_t, uint32_t> get_qa_length() {
    if (!has_data()) {
        throw std::out_of_range("Request trace has no unread rows");
    }
    const auto& row = table[row_index++];
    return std::make_pair(row[0], row[answer_index]);
}

void parse(std::string path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        throw std::runtime_error("Cannot open request trace file '" + path + "'");
    }

    std::string line;
    if (!std::getline(input_file, line)) {
        throw std::invalid_argument("Request trace file is empty: '" + path + "'");
    }
    std::istringstream header_stream(line);
    std::string column_name;
    while (std::getline(header_stream, column_name, ',')) {
        columns.push_back(column_name);
    }
    if (columns.empty() || answer_index >= columns.size()) {
        throw std::invalid_argument(
            "Request trace '" + path + "' does not contain answer column " +
            std::to_string(answer_index));
    }

    uint32_t line_number = 1;
    while (std::getline(input_file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        std::vector<uint32_t> buffer;
        std::istringstream iss(line);
        std::string cell;
        while (std::getline(iss, cell, ',')) {
            try {
                std::size_t parsed_characters = 0;
                const unsigned long value = std::stoul(cell, &parsed_characters);
                if (parsed_characters != cell.size() ||
                    value > std::numeric_limits<uint32_t>::max()) {
                    throw std::out_of_range("not a uint32 value");
                }
                buffer.push_back(static_cast<uint32_t>(value));
            } catch (const std::exception&) {
                throw std::invalid_argument(
                    "Invalid request trace value at '" + path + "':" +
                    std::to_string(line_number) + ", column " +
                    std::to_string(buffer.size() + 1) + ": '" + cell + "'");
            }
        }
        if (buffer.size() != columns.size()) {
            throw std::invalid_argument(
                "Request trace row width mismatch at '" + path + "':" +
                std::to_string(line_number) + "; expected " +
                std::to_string(columns.size()) + " columns, found " +
                std::to_string(buffer.size()));
        }
        table.push_back(buffer);
    }
    if (table.empty()) {
        throw std::invalid_argument(
            "Request trace contains no data rows: '" + path + "'");
    }
}
}  // namespace RequestGenerator
