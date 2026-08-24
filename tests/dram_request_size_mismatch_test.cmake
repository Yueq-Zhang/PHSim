cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(output_dir "${BINARY_DIR}/test-output/dram-request-size-mismatch")
file(REMOVE_RECURSE "${output_dir}")
file(MAKE_DIRECTORY "${output_dir}")

execute_process(
    COMMAND
        "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_burst_mismatch.json"
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
        "Simulator accepted inconsistent dram_req_size; see ${output_dir}/test-process.log")
endif()

set(combined_output "${simulator_stdout}\n${simulator_stderr}")
if(NOT combined_output MATCHES "DRAM request-size mismatch" OR
   NOT combined_output MATCHES "dram_req_size=64 bytes" OR
   NOT combined_output MATCHES "= 32 bytes")
    message(FATAL_ERROR
        "Mismatch diagnostic did not report configured and derived sizes; "
        "see ${output_dir}/test-process.log")
endif()

message(STATUS
    "DRAM request-size mismatch was rejected with configured=64 B and derived=32 B")
