cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(output_dir "${BINARY_DIR}/test-output/dram-frequency-mismatch")
file(REMOVE_RECURSE "${output_dir}")
file(MAKE_DIRECTORY "${output_dir}")

execute_process(
    COMMAND
        "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_frequency_mismatch.json"
        --output_path
        "${output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE simulator_result
    OUTPUT_VARIABLE simulator_stdout
    ERROR_VARIABLE simulator_stderr
    TIMEOUT 20
)

file(WRITE "${output_dir}/test-process.log"
    "${simulator_stdout}\n${simulator_stderr}")

if(simulator_result EQUAL 0)
    message(FATAL_ERROR
        "Simulator accepted inconsistent dram_freq/tCK; see ${output_dir}/test-process.log")
endif()

set(combined_output "${simulator_stdout}\n${simulator_stderr}")
if(NOT combined_output MATCHES "DRAM frequency mismatch" OR
   NOT combined_output MATCHES "dram_freq=1000 MHz" OR
   NOT combined_output MATCHES "tCK=1.25 ns" OR
   NOT combined_output MATCHES "800 MHz")
    message(FATAL_ERROR
        "Frequency mismatch diagnostic did not report all values; "
        "see ${output_dir}/test-process.log")
endif()

message(STATUS
    "DRAM frequency mismatch was rejected with PIM=1000 MHz and tCK-derived=800 MHz")
