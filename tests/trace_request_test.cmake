cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(output_dir "${BINARY_DIR}/test-output/trace-request")
file(REMOVE_RECURSE "${output_dir}")
file(MAKE_DIRECTORY "${output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_trace_event_driven.json"
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
   NOT simulator_stdout MATCHES "parsed 1 lines" OR
   NOT simulator_stdout MATCHES
       "Request #0 is generated.*input size:4, output size:7" OR
   simulator_stdout MATCHES "input size:99|output size:99" OR
   simulator_stdout MATCHES "Invalid batch size: 99" OR
   NOT simulator_stdout MATCHES
       "Trace mode effective request count: 1.*gen_request_count=99 is ignored" OR
   NOT simulator_stdout MATCHES
       "weight_interleave_columns set to 8.*batch size 1" OR
   NOT simulator_stdout MATCHES "Finish the simulation")
    message(FATAL_ERROR
        "Trace-driven request did not preserve CSV input/output sizes; see "
        "${output_dir}/test-process.log")
endif()

message(STATUS
    "Trace-driven request used one CSV request with input/output sizes 4/7")
