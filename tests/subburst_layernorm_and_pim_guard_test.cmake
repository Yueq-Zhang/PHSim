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

set(channel_aligned_split_output_dir
    "${BINARY_DIR}/test-output/layernorm-channel-aligned-split")
file(REMOVE_RECURSE "${channel_aligned_split_output_dir}")
file(MAKE_DIRECTORY "${channel_aligned_split_output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_layernorm_channel_split_event_driven.json"
        --output_path "${channel_aligned_split_output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE channel_aligned_split_result
    OUTPUT_VARIABLE channel_aligned_split_stdout
    ERROR_VARIABLE channel_aligned_split_stderr
    TIMEOUT 20
)
file(WRITE "${channel_aligned_split_output_dir}/test-process.log"
    "${channel_aligned_split_stdout}\n${channel_aligned_split_stderr}")

if(NOT channel_aligned_split_result EQUAL 0 OR
   NOT channel_aligned_split_stdout MATCHES "Finish the simulation" OR
   NOT channel_aligned_split_stdout MATCHES
       "Outstanding MemoryAccess requests at simulation end: 0")
    message(FATAL_ERROR
        "Channel-aligned LayerNorm split did not complete; see "
        "${channel_aligned_split_output_dir}/test-process.log")
endif()
set(channel_aligned_split_output
    "${channel_aligned_split_stdout}\n${channel_aligned_split_stderr}")
if(NOT channel_aligned_split_output MATCHES
       "LayerNorm for 1 batches, inner loop: \\[4\\], outer loop: \\[2\\]")
    message(FATAL_ERROR
        "LayerNorm did not keep its forced split aligned to the two-channel "
        "row group; see "
        "${channel_aligned_split_output_dir}/test-process.log")
endif()
if(NOT channel_aligned_split_output MATCHES
       "LayerNorm cycle 30")
    message(FATAL_ERROR
        "Single-VU LayerNorm instructions overlapped instead of queueing "
        "serially; see "
        "${channel_aligned_split_output_dir}/test-process.log")
endif()
if(NOT EXISTS "${channel_aligned_split_output_dir}/request_stats.tsv")
    message(FATAL_ERROR
        "Channel-aligned LayerNorm split did not produce request statistics")
endif()
file(STRINGS "${channel_aligned_split_output_dir}/request_stats.tsv"
    channel_aligned_request_lines)
list(LENGTH channel_aligned_request_lines channel_aligned_request_line_count)
if(NOT channel_aligned_request_line_count EQUAL 2)
    message(FATAL_ERROR
        "Channel-aligned LayerNorm split should complete exactly one request")
endif()

set(two_vu_output_dir
    "${BINARY_DIR}/test-output/layernorm-two-vu")
file(REMOVE_RECURSE "${two_vu_output_dir}")
file(MAKE_DIRECTORY "${two_vu_output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_layernorm_two_vu_event_driven.json"
        --output_path "${two_vu_output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE two_vu_result
    OUTPUT_VARIABLE two_vu_stdout
    ERROR_VARIABLE two_vu_stderr
    TIMEOUT 20
)
file(WRITE "${two_vu_output_dir}/test-process.log"
    "${two_vu_stdout}\n${two_vu_stderr}")

if(NOT two_vu_result EQUAL 0 OR
   NOT two_vu_stdout MATCHES "Finish the simulation" OR
   NOT two_vu_stdout MATCHES
       "Outstanding MemoryAccess requests at simulation end: 0")
    message(FATAL_ERROR
        "Two-VU LayerNorm did not complete; see "
        "${two_vu_output_dir}/test-process.log")
endif()
set(two_vu_output "${two_vu_stdout}\n${two_vu_stderr}")
if(NOT two_vu_output MATCHES
       "LayerNorm for 1 batches, inner loop: \\[4\\], outer loop: \\[2\\]")
    message(FATAL_ERROR
        "Two-VU LayerNorm changed the channel-aligned tile layout; see "
        "${two_vu_output_dir}/test-process.log")
endif()
if(NOT two_vu_output MATCHES "LayerNorm cycle 30")
    message(FATAL_ERROR
        "Two-VU LayerNorm did not preserve aggregate vector-unit cycle "
        "accounting; see ${two_vu_output_dir}/test-process.log")
endif()
if(NOT EXISTS "${two_vu_output_dir}/request_stats.tsv")
    message(FATAL_ERROR
        "Two-VU LayerNorm did not produce request statistics")
endif()
file(STRINGS "${two_vu_output_dir}/request_stats.tsv" two_vu_request_lines)
list(LENGTH two_vu_request_lines two_vu_request_line_count)
if(NOT two_vu_request_line_count EQUAL 2)
    message(FATAL_ERROR
        "Two-VU LayerNorm should complete exactly one request")
endif()

set(undersized_spad_output_dir
    "${BINARY_DIR}/test-output/layernorm-minimum-tile-overflow")
file(REMOVE_RECURSE "${undersized_spad_output_dir}")
file(MAKE_DIRECTORY "${undersized_spad_output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_layernorm_minimum_tile_overflow_event_driven.json"
        --output_path "${undersized_spad_output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE undersized_spad_result
    OUTPUT_VARIABLE undersized_spad_stdout
    ERROR_VARIABLE undersized_spad_stderr
    TIMEOUT 20
)
file(WRITE "${undersized_spad_output_dir}/test-process.log"
    "${undersized_spad_stdout}\n${undersized_spad_stderr}")

if(undersized_spad_result EQUAL 0)
    message(FATAL_ERROR
        "LayerNorm accepted a minimum tile that exceeds SPAD capacity")
endif()
set(undersized_spad_output
    "${undersized_spad_stdout}\n${undersized_spad_stderr}")
if(NOT undersized_spad_output MATCHES
       "LayerNorm.*minimum tile cannot fit in SRAM" OR
   NOT undersized_spad_output MATCHES
       "SPAD requires 1088 bytes but provides 512 bytes" OR
   NOT undersized_spad_output MATCHES
       "Accum SPAD requires 2048 bytes but provides 524288 bytes" OR
   NOT undersized_spad_output MATCHES "row_alignment=2")
    message(FATAL_ERROR
        "LayerNorm minimum-tile diagnostic did not distinguish the SPAD "
        "capacity; see ${undersized_spad_output_dir}/test-process.log")
endif()

set(undersized_accum_spad_output_dir
    "${BINARY_DIR}/test-output/layernorm-accum-minimum-tile-overflow")
file(REMOVE_RECURSE "${undersized_accum_spad_output_dir}")
file(MAKE_DIRECTORY "${undersized_accum_spad_output_dir}")
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_layernorm_accum_minimum_tile_overflow_event_driven.json"
        --output_path "${undersized_accum_spad_output_dir}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE undersized_accum_spad_result
    OUTPUT_VARIABLE undersized_accum_spad_stdout
    ERROR_VARIABLE undersized_accum_spad_stderr
    TIMEOUT 20
)
file(WRITE "${undersized_accum_spad_output_dir}/test-process.log"
    "${undersized_accum_spad_stdout}\n${undersized_accum_spad_stderr}")

if(undersized_accum_spad_result EQUAL 0)
    message(FATAL_ERROR
        "LayerNorm accepted a minimum tile that exceeds Accum SPAD capacity")
endif()
set(undersized_accum_spad_output
    "${undersized_accum_spad_stdout}\n${undersized_accum_spad_stderr}")
if(NOT undersized_accum_spad_output MATCHES
       "LayerNorm.*minimum tile cannot fit in SRAM" OR
   NOT undersized_accum_spad_output MATCHES
       "SPAD requires 1088 bytes but provides 524288 bytes" OR
   NOT undersized_accum_spad_output MATCHES
       "Accum SPAD requires 2048 bytes but provides 512 bytes" OR
   NOT undersized_accum_spad_output MATCHES "row_alignment=2")
    message(FATAL_ERROR
        "LayerNorm minimum-tile diagnostic did not distinguish the Accum "
        "SPAD capacity; see "
        "${undersized_accum_spad_output_dir}/test-process.log")
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
if(invalid_pim_output MATCHES "Assertion.*failed")
    message(FATAL_ERROR
        "Undersized PIM Decode reached an internal assertion before the "
        "configuration diagnostic; see "
        "${invalid_pim_output_dir}/test-process.log")
endif()

message(STATUS
    "Sub-burst H16 LayerNorm completed, a forced split stayed channel-row "
    "aligned for one and two vector units, independent SPAD and Accum SPAD "
    "minimum-tile overflows were rejected, and undersized PIM Decode was "
    "rejected")
