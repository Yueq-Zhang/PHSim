cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(output_dir "${BINARY_DIR}/test-output/dram-channel-mismatch")
file(REMOVE_RECURSE "${output_dir}")
file(MAKE_DIRECTORY "${output_dir}")

execute_process(
    COMMAND
        "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_channel_mismatch.json"
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
        "Simulator accepted inconsistent channel counts; see ${output_dir}/test-process.log")
endif()

set(combined_output "${simulator_stdout}\n${simulator_stderr}")
if(NOT combined_output MATCHES "DRAM channel-count mismatch" OR
   NOT combined_output MATCHES "dram_channels=4" OR
   NOT combined_output MATCHES "channels=2")
    message(FATAL_ERROR
        "Channel mismatch diagnostic did not report both values; "
        "see ${output_dir}/test-process.log")
endif()

message(STATUS
    "DRAM channel mismatch was rejected with PIM=4 and memory=2")
