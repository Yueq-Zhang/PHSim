cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(output_dir "${BINARY_DIR}/test-output/max-active-requests")
file(REMOVE_RECURSE "${output_dir}")
file(MAKE_DIRECTORY "${output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_max_active_requests_event_driven.json"
        --output_path "${output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE simulator_result
    OUTPUT_VARIABLE simulator_stdout
    ERROR_VARIABLE simulator_stderr
    TIMEOUT 20
)
file(WRITE "${output_dir}/test-process.log"
    "${simulator_stdout}\n${simulator_stderr}")

if(NOT simulator_result EQUAL 0 OR
   NOT simulator_stdout MATCHES "Finish the simulation")
    message(FATAL_ERROR
        "max_active_reqs simulation failed; see ${output_dir}/test-process.log")
endif()

string(REGEX MATCHALL
    "Scheduler admitted request [01]: 1/1 resident requests"
    admissions "${simulator_stdout}")
list(LENGTH admissions admission_count)
string(REGEX MATCHALL "Client Receive response From Scheduler!"
    responses "${simulator_stdout}")
list(LENGTH responses response_count)
string(FIND "${simulator_stdout}"
    "Scheduler admitted request 0: 1/1 resident requests" request0_position)
string(FIND "${simulator_stdout}"
    "Scheduler:: The inference process of request 0 is done"
    request0_completion_position)
string(FIND "${simulator_stdout}"
    "Scheduler admitted request 1: 1/1 resident requests" request1_position)

if(NOT admission_count EQUAL 2 OR
   NOT response_count EQUAL 2 OR
   request0_position LESS 0 OR
   request0_completion_position LESS request0_position OR
   request1_position LESS request0_completion_position OR
   simulator_stdout MATCHES "[2-9]/1 resident requests")
    message(FATAL_ERROR
        "max_active_reqs=1 did not serialize the two requests; see "
        "${output_dir}/test-process.log")
endif()

message(STATUS "max_active_reqs=1 serialized two generated requests")
