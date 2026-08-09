#include <iostream>
#include <limits>
#include <fstream>
#include <chrono>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "../ext/argparse.hpp"
#include "../ext/json.hpp"

#include "simulator.hpp"
#include "common_function.hpp"


int finish_trans;
std::ofstream callback_record_file("../output/NMC_callback_record.txt");
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

    nlohmann::json simulation_config = load_config(simulation_config_path);

    // load the config files
    compute_die_config_path = simulation_config["compute_die_config_file_path"];
    memory_config_path = simulation_config["DRAM_config_file_path"];
    pim_config_path = simulation_config["PIM_config_file_path"];
    inference_config_path= simulation_config["inference_config_file_path"];
    model_config_path= simulation_config["model_config_file_path"];;
    request_trace_file_path = simulation_config["request_file_path"];

    // create the log file
    try {
        std::string log_file_path = output_path + "/log.txt";
        if (std::filesystem::exists(log_file_path)) {
            if (!std::filesystem::remove(log_file_path)) {
                throw std::runtime_error("Removing the exist log file " + log_file_path + " Failed" );
            }
            std::cout << "Remove the exist log file" << std::endl;
        }
        file_logger = spdlog::basic_logger_mt("logger",  log_file_path);
        file_logger->set_level(spdlog::level::trace);
        // spdlog::set_default_logger(file_logger);
    } catch (const spdlog::spdlog_ex& ex) {
        throw std::runtime_error("Log init failed: " + std::string(ex.what()));
    }
    Config::system_config.initialize_from_config_path(compute_die_config_path, memory_config_path, pim_config_path, inference_config_path, model_config_path, request_trace_file_path, output_path);

    PIM_Parameters::init(Config::system_config);  // Initial global PIM configuration
    MyAddressAllocator::init(Config::system_config); // MyAddress Allocator

    if (Config::system_config.dram_data_container_enable) {
        DRAMDataContainer::init(Config::system_config);
    }

    if (Config::system_config.virtual_mem_hash_enable) {
        TwoLevelPageMapper::init_two_level_mapper();
    }

    // Model
    std::string model_name = Config::system_config.model_name;
    auto model = std::make_shared<Model>(Config::system_config, model_name);

    // Simulator structure
    auto simulator = std::make_unique<Simulator>(Config::system_config);

    printf("Launching model\n");
    simulator->launch_model(model);  // Add model structure to the simulator for simulation
    spdlog::info("Launch model: {}", model_name);
    simulator->run(model_name);

    std::cout << "Finish the simulation" << std::endl;

    simulator.reset();
    model.reset();
    MyAddressAllocator::cleanup();
    DRAMDataContainer::cleanup();

    file_logger->info("Finish the simulation");
    file_logger->flush();
    file_logger.reset();

    return 0;
}
