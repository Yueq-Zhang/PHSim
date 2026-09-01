cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(prefill_output_dir
    "${BINARY_DIR}/test-output/subburst-layernorm-prefill")
file(REMOVE_RECURSE "${prefill_output_dir}")
file(MAKE_DIRECTORY "${prefill_output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_subburst_layernorm_event_driven.json"
        --output_path "${prefill_output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE prefill_result
    OUTPUT_VARIABLE prefill_stdout
    ERROR_VARIABLE prefill_stderr
    TIMEOUT 20
)
file(WRITE "${prefill_output_dir}/test-process.log"
    "${prefill_stdout}\n${prefill_stderr}")

if(NOT prefill_result EQUAL 0 OR
   NOT prefill_stdout MATCHES "Finish the simulation" OR
   NOT prefill_stdout MATCHES
       "Outstanding MemoryAccess requests at simulation end: 0")
    message(FATAL_ERROR
        "Sub-burst H16 LayerNorm Prefill did not complete; see "
        "${prefill_output_dir}/test-process.log")
endif()
if(prefill_stdout MATCHES "Current load instruction has no src address")
    message(FATAL_ERROR
        "Sub-burst H16 LayerNorm still emitted an empty-address weight load")
endif()
if(NOT EXISTS "${prefill_output_dir}/request_stats.tsv")
    message(FATAL_ERROR
        "Sub-burst H16 LayerNorm did not produce request statistics")
endif()
file(STRINGS "${prefill_output_dir}/request_stats.tsv" request_lines)
list(LENGTH request_lines request_line_count)
if(NOT request_line_count EQUAL 3)
    message(FATAL_ERROR
        "Sub-burst H16 LayerNorm should complete exactly two requests")
endif()

set(invalid_pim_output_dir
    "${BINARY_DIR}/test-output/invalid-pim-decode-h32")
file(REMOVE_RECURSE "${invalid_pim_output_dir}")
file(MAKE_DIRECTORY "${invalid_pim_output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_invalid_pim_decode_event_driven.json"
        --output_path "${invalid_pim_output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE invalid_pim_result
    OUTPUT_VARIABLE invalid_pim_stdout
    ERROR_VARIABLE invalid_pim_stderr
    TIMEOUT 20
)
file(WRITE "${invalid_pim_output_dir}/test-process.log"
    "${invalid_pim_stdout}\n${invalid_pim_stderr}")

if(invalid_pim_result EQUAL 0)
    message(FATAL_ERROR
        "Undersized H32 PIM Decode configuration was accepted")
endif()
set(invalid_pim_output "${invalid_pim_stdout}\n${invalid_pim_stderr}")
if(NOT invalid_pim_output MATCHES
   "PIM Decode weight output dimension 32 is too small" OR
   NOT invalid_pim_output MATCHES
   "require at least 256 output elements")
    message(FATAL_ERROR
        "Undersized PIM Decode diagnostic omitted the invalid and required "
        "dimensions; see ${invalid_pim_output_dir}/test-process.log")
endif()

message(STATUS
    "Sub-burst H16 LayerNorm completed and undersized PIM Decode was rejected")
