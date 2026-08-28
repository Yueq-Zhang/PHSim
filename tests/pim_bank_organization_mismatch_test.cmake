cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(output_dir "${BINARY_DIR}/test-output/pim-bank-organization-mismatch")
file(REMOVE_RECURSE "${output_dir}")
file(MAKE_DIRECTORY "${output_dir}")

execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_dual_bank_mismatch.json"
        --output_path "${output_dir}"
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
        "Simulator accepted dual_bank=true with pim_type=SINGLE")
endif()

set(combined_output "${simulator_stdout}\n${simulator_stderr}")
if(NOT combined_output MATCHES "PIM bank-organization mismatch" OR
   NOT combined_output MATCHES "dual_bank=true" OR
   NOT combined_output MATCHES "pim_type=SINGLE")
    message(FATAL_ERROR
        "PIM organization diagnostic omitted the conflicting values; see "
        "${output_dir}/test-process.log")
endif()

message(STATUS "PIM dual_bank/pim_type mismatch was rejected")
