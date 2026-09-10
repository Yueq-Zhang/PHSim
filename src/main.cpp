#include <iostream>
#include <limits>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <set>
#include <vector>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "../ext/argparse.hpp"
#include "../ext/json.hpp"

#include "simulator.hpp"
#include "common_function.hpp"
#include "Client/RequestGenerator.h"


int finish_trans;
std::chrono::time_point<std::chrono::system_clock>  time_start = std::chrono::high_resolution_clock::now();
std::chrono::time_point<std::chrono::system_clock>  time_end = std::chrono::high_resolution_clock::now();  // record the end time
int record_interval = 200;


int main(int argc, char *argv[]) {

    std::string simulation_config_path;
    std::string compute_die_config_path;
    std::string memory_config_path;
    std::string pim_config_path;
    std::string inference_config_path;
    std::string model_config_path;
    std::string request_trace_file_path;
    std::string trace_file_path;
    std::string output_path;

    std::shared_ptr<spdlog::logger> file_logger;

    argparse::ArgumentParser parser("NMC_Simulator");
    parser.add_argument("-sc", "--simulation_config").help("Path of the entire simulation config file, for the NPU-PIM system initialization and simulation")
    .default_value("").store_into(simulation_config_path);
    parser.add_argument("-o", "--output_path").help("Path of the generated output files").default_value("output").store_into(output_path);

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    try {
    nlohmann::json simulation_config = load_config(simulation_config_path);
    using phsim::JsonValueKind;
    phsim::ConfigValidator::ValidateJsonObject(
        simulation_config, "simulation", simulation_config_path,
        {
            {"compute_die_config_file_path", JsonValueKind::String, true},
            {"DRAM_config_file_path", JsonValueKind::String, true},
            {"PIM_config_file_path", JsonValueKind::String, true},
            {"model_config_file_path", JsonValueKind::String, true},
            {"inference_config_file_path", JsonValueKind::String, true},
            {"request_file_path", JsonValueKind::String, true},
        });

    // load the config files
    const auto resolve_reference =
        [&simulation_config_path](const std::string& configured_path,
                                  const char* field) {
            const auto resolved = phsim::ResolveConfigPath(
                configured_path, simulation_config_path);
            if (resolved.used_legacy_working_directory) {
                spdlog::warn(
                    "{} in '{}' uses deprecated working-directory-relative "
                    "resolution; rewrite it relative to the simulation "
                    "configuration file",
                    field, simulation_config_path);
            }
            return resolved.path.string();
        };

    compute_die_config_path = resolve_reference(
        simulation_config["compute_die_config_file_path"].get<std::string>(),
        "compute_die_config_file_path");
    memory_config_path = resolve_reference(
        simulation_config["DRAM_config_file_path"].get<std::string>(),
        "DRAM_config_file_path");
    pim_config_path = resolve_reference(
        simulation_config["PIM_config_file_path"].get<std::string>(),
        "PIM_config_file_path");
    inference_config_path = resolve_reference(
        simulation_config["inference_config_file_path"].get<std::string>(),
        "inference_config_file_path");
    model_config_path = resolve_reference(
        simulation_config["model_config_file_path"].get<std::string>(),
        "model_config_file_path");
    request_trace_file_path = resolve_reference(
        simulation_config["request_file_path"].get<std::string>(),
        "request_file_path");

    for (const auto& config_path : {
             std::pair<const char*, const std::string*>{
                 "compute_die_config_file_path", &compute_die_config_path},
             {"DRAM_config_file_path", &memory_config_path},
             {"PIM_config_file_path", &pim_config_path},
             {"inference_config_file_path", &inference_config_path},
             {"model_config_file_path", &model_config_path}}) {
        if (config_path.second->empty()) {
            throw std::invalid_argument(
                std::string(config_path.first) +
                " must not be empty in simulation config '" +
                simulation_config_path + "'");
        }
    }

    if (output_path.empty()) {
        throw std::invalid_argument("output_path must not be empty");
    }
    std::error_code directory_error;
    std::filesystem::create_directories(output_path, directory_error);
    if (directory_error || !std::filesystem::is_directory(output_path)) {
        throw std::runtime_error(
            "Cannot create output directory '" + output_path + "': " +
            directory_error.message());
    }

    const nlohmann::json logging_config = load_config(compute_die_config_path);
    std::string configured_log_level = "info";
    if (const auto log_level = logging_config.find("log_level");
        log_level != logging_config.end()) {
        if (!log_level->is_string()) {
            throw std::invalid_argument(
                "Field 'log_level' in compute config '" +
                compute_die_config_path + "' must be a string");
        }
        configured_log_level = log_level->get<std::string>();
    }
    static const std::set<std::string> supported_log_levels = {
        "trace", "debug", "info", "warn", "error", "critical", "off"};
    if (supported_log_levels.count(configured_log_level) == 0) {
        throw std::invalid_argument(
            "Unsupported log_level='" + configured_log_level +
            "' in compute config '" + compute_die_config_path + "'");
    }

    // create the log file
    try {
        std::string log_file_path = output_path + "/log.txt";
        auto console_sink =
            std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            log_file_path, true);
        std::vector<spdlog::sink_ptr> sinks = {console_sink, file_sink};
        file_logger = std::make_shared<spdlog::logger>(
            "phsim", sinks.begin(), sinks.end());
        file_logger->set_level(spdlog::level::from_str(configured_log_level));
        file_logger->flush_on(spdlog::level::warn);
        file_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        spdlog::set_default_logger(file_logger);
    } catch (const spdlog::spdlog_ex& ex) {
        throw std::runtime_error("Log init failed: " + std::string(ex.what()));
    }
    Config::system_config.initialize_from_config_path(compute_die_config_path, memory_config_path, pim_config_path, inference_config_path, model_config_path, request_trace_file_path, output_path);

    if (!Config::system_config.gen_request) {
        constexpr uint32_t answer_index = 1;
        RequestGenerator::init(
            Config::system_config.request_dataset_path, answer_index);
        Config::system_config.effective_request_count =
            static_cast<uint32_t>(RequestGenerator::get_total_req_cnt());
        spdlog::info(
            "Trace mode effective request count: {} (gen_request_count={} "
            "is ignored)",
            Config::system_config.effective_request_count,
            Config::system_config.gen_request_count);
    }

    PIM_Parameters::init(Config::system_config);  // Initial global PIM configuration
    MyAddressAllocator::init(Config::system_config); // MyAddress Allocator

    if (Config::system_config.virtual_mem_hash_enable) {
        TwoLevelPageMapper::init_two_level_mapper();
    }

    // Model
    std::string model_name = Config::system_config.model_name;
    auto model = std::make_shared<Model>(Config::system_config, model_name);

    // Simulator structure
    auto simulator = std::make_unique<Simulator>(Config::system_config);

    spdlog::info("Launching model");
    simulator->launch_model(model);  // Add model structure to the simulator for simulation
    spdlog::info("Launch model: {}", model_name);
    simulator->run(model_name);

    spdlog::info("Finish the simulation");

    simulator.reset();
    model.reset();
    TwoLevelPageMapper::cleanup_two_level_mapper();
    MyAddressAllocator::cleanup();
    spdlog::default_logger()->flush();
    file_logger.reset();
    spdlog::shutdown();

    return 0;
    } catch (const std::exception& error) {
        spdlog::critical("Simulation failed: {}", error.what());
        if (spdlog::default_logger()) {
            spdlog::default_logger()->flush();
        }
        return 1;
    }
}
