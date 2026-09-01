cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR TEST_MODE)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

function(run_backend backend config_name output_variable)
    set(output_dir
        "${BINARY_DIR}/test-output/${TEST_MODE}/${backend}")
    file(REMOVE_RECURSE "${output_dir}")
    file(MAKE_DIRECTORY "${output_dir}")

    execute_process(
        COMMAND
            "${SIMULATOR}"
            --simulation_config
            "${SOURCE_DIR}/tests/fixtures/smoke/${config_name}"
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

    if(NOT simulator_result EQUAL 0)
        message(FATAL_ERROR
            "${backend} simulation failed with exit code ${simulator_result}. "
            "See ${output_dir}/test-process.log")
    endif()
    if(NOT simulator_stdout MATCHES "Finish the simulation")
        message(FATAL_ERROR
            "${backend} simulation did not report normal completion")
    endif()
    if(NOT simulator_stdout MATCHES
       "Outstanding MemoryAccess requests at simulation end: 0")
        message(FATAL_ERROR
            "${backend} simulation did not release every MemoryAccess")
    endif()

    foreach(required_file IN ITEMS _summary.tsv icnt_traffic.json)
        if(NOT EXISTS "${output_dir}/${required_file}")
            message(FATAL_ERROR
                "${backend} simulation did not produce ${required_file}")
        endif()
    endforeach()

    file(STRINGS "${output_dir}/_summary.tsv" summary_lines)
    list(LENGTH summary_lines summary_line_count)
    if(NOT summary_line_count EQUAL 2)
        message(FATAL_ERROR
            "${backend} summary should contain one header and one stage row")
    endif()
    list(GET summary_lines 1 summary_row)
    string(REPLACE "\t" ";" summary_fields "${summary_row}")
    list(GET summary_fields 0 stage_name)
    list(GET summary_fields 1 total_cycles)
    list(GET summary_fields 3 memory_bandwidth)
    if(NOT stage_name STREQUAL "Single_test")
        message(FATAL_ERROR
            "${backend} completed unexpected stage: ${stage_name}")
    endif()
    if(NOT total_cycles MATCHES "^[0-9]+$" OR total_cycles LESS_EQUAL 0)
        message(FATAL_ERROR
            "${backend} produced invalid total cycle count: ${total_cycles}")
    endif()
    if(NOT memory_bandwidth MATCHES "^[0-9]+([.][0-9]+)?$")
        message(FATAL_ERROR
            "${backend} produced invalid memory bandwidth: ${memory_bandwidth}")
    endif()

    set(${output_variable} "${output_dir}" PARENT_SCOPE)
endfunction()

function(run_iterative_decode_backend backend config_name output_variable)
    set(output_dir
        "${BINARY_DIR}/test-output/${TEST_MODE}/${backend}")
    file(REMOVE_RECURSE "${output_dir}")
    file(MAKE_DIRECTORY "${output_dir}")

    execute_process(
        COMMAND
            "${SIMULATOR}"
            --simulation_config
            "${SOURCE_DIR}/tests/fixtures/smoke/${config_name}"
            --output_path
            "${output_dir}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE simulator_result
        OUTPUT_VARIABLE simulator_stdout
        ERROR_VARIABLE simulator_stderr
        TIMEOUT 60
    )

    file(WRITE "${output_dir}/test-process.log"
        "${simulator_stdout}\n${simulator_stderr}")

    if(NOT simulator_result EQUAL 0)
        message(FATAL_ERROR
            "${backend} iterative Decode failed with exit code "
            "${simulator_result}. See ${output_dir}/test-process.log")
    endif()
    if(NOT simulator_stdout MATCHES "Finish the simulation" OR
       NOT simulator_stdout MATCHES
           "Outstanding MemoryAccess requests at simulation end: 0")
        message(FATAL_ERROR
            "${backend} iterative Decode did not finish cleanly")
    endif()

    string(REGEX MATCHALL
        "Scheduler:: Request 0 generated token [12]/2"
        token_progress "${simulator_stdout}")
    list(LENGTH token_progress token_progress_count)
    if(NOT token_progress_count EQUAL 2 OR
       NOT simulator_stdout MATCHES "generated token 1/2" OR
       NOT simulator_stdout MATCHES "generated token 2/2")
        message(FATAL_ERROR
            "${backend} did not execute exactly two output-token iterations")
    endif()

    string(REGEX MATCHALL "New Program for PIM" pim_decode_programs
        "${simulator_stdout}")
    list(LENGTH pim_decode_programs pim_decode_program_count)
    if(NOT pim_decode_program_count EQUAL 2)
        message(FATAL_ERROR
            "${backend} expected two PIM Decode programs, found "
            "${pim_decode_program_count}")
    endif()

    file(STRINGS "${output_dir}/_summary.tsv" summary_lines)
    list(LENGTH summary_lines summary_line_count)
    if(NOT summary_line_count EQUAL 4)
        message(FATAL_ERROR
            "${backend} expected header + Prefill + two Decode rows, found "
            "${summary_line_count} lines")
    endif()
    list(GET summary_lines 1 prefill_row)
    list(GET summary_lines 2 decode_row_1)
    list(GET summary_lines 3 decode_row_2)
    if(NOT prefill_row MATCHES "^Prefill\t" OR
       NOT decode_row_1 MATCHES "^Decode\t" OR
       NOT decode_row_2 MATCHES "^Decode\t")
        message(FATAL_ERROR
            "${backend} iterative stage order is not Prefill/Decode/Decode")
    endif()
    set(${output_variable} "${output_dir}" PARENT_SCOPE)
endfunction()

function(run_continuous_batch_backend backend config_name output_variable)
    set(output_dir
        "${BINARY_DIR}/test-output/${TEST_MODE}/${backend}")
    file(REMOVE_RECURSE "${output_dir}")
    file(MAKE_DIRECTORY "${output_dir}")

    execute_process(
        COMMAND
            "${SIMULATOR}"
            --simulation_config
            "${SOURCE_DIR}/tests/fixtures/smoke/${config_name}"
            --output_path
            "${output_dir}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE simulator_result
        OUTPUT_VARIABLE simulator_stdout
        ERROR_VARIABLE simulator_stderr
        TIMEOUT 60
    )

    file(WRITE "${output_dir}/test-process.log"
        "${simulator_stdout}\n${simulator_stderr}")
    if(NOT simulator_result EQUAL 0 OR
       NOT simulator_stdout MATCHES "Finish the simulation" OR
       NOT simulator_stdout MATCHES
           "Outstanding MemoryAccess requests at simulation end: 0")
        message(FATAL_ERROR
            "${backend} continuous batching did not finish cleanly")
    endif()

    foreach(expected_batch IN ITEMS
            "Continuous batch 0 formed: stage=Prefill, requests=[0,1]"
            "Continuous batch 1 formed: stage=Decode, requests=[0,1]"
            "Continuous batch 2 formed: stage=Prefill, requests=[2]"
            "Continuous batch 3 formed: stage=Decode, requests=[0,2]"
            "Continuous batch 4 formed: stage=Decode, requests=[0,2]")
        string(FIND "${simulator_stdout}" "${expected_batch}"
            expected_batch_position)
        if(expected_batch_position EQUAL -1)
            message(FATAL_ERROR
                "${backend} missing expected scheduling event: "
                "${expected_batch}")
        endif()
    endforeach()

    foreach(request_id IN ITEMS 0 1 2)
        if(request_id EQUAL 0)
            set(expected_tokens 3)
        elseif(request_id EQUAL 1)
            set(expected_tokens 1)
        else()
            set(expected_tokens 2)
        endif()
        string(REGEX MATCHALL
            "Scheduler:: Request ${request_id} generated token [0-9]+/${expected_tokens}"
            token_progress "${simulator_stdout}")
        list(LENGTH token_progress generated_token_count)
        if(NOT generated_token_count EQUAL expected_tokens)
            message(FATAL_ERROR
                "${backend} request ${request_id} generated "
                "${generated_token_count}/${expected_tokens} tokens")
        endif()
    endforeach()

    string(REGEX MATCHALL "Client Receive response From Scheduler!"
        client_responses "${simulator_stdout}")
    list(LENGTH client_responses client_response_count)
    if(NOT client_response_count EQUAL 3)
        message(FATAL_ERROR
            "${backend} expected three completed client requests, found "
            "${client_response_count}")
    endif()

    string(FIND "${simulator_stdout}"
        "Scheduler:: Request 1 completed after generating 1/1 tokens"
        short_request_completion)
    string(FIND "${simulator_stdout}"
        "Scheduler admitted request 2: 2/2 resident requests"
        replacement_request_admission)
    if(short_request_completion EQUAL -1 OR
       replacement_request_admission EQUAL -1 OR
       replacement_request_admission LESS short_request_completion)
        message(FATAL_ERROR
            "${backend} did not admit request 2 after request 1 released "
            "its active slot")
    endif()

    file(STRINGS "${output_dir}/_summary.tsv" summary_lines)
    list(LENGTH summary_lines summary_line_count)
    if(NOT summary_line_count EQUAL 6)
        message(FATAL_ERROR
            "${backend} expected header + five continuous batches, found "
            "${summary_line_count} lines")
    endif()
    set(expected_stages Prefill Decode Prefill Decode Decode)
    foreach(summary_index RANGE 1 5)
        list(GET summary_lines ${summary_index} summary_row)
        math(EXPR stage_index "${summary_index} - 1")
        list(GET expected_stages ${stage_index} expected_stage)
        if(NOT summary_row MATCHES "^${expected_stage}\t")
            message(FATAL_ERROR
                "${backend} continuous stage ${summary_index} is not "
                "${expected_stage}: ${summary_row}")
        endif()
    endforeach()

    set(${output_variable} "${output_dir}" PARENT_SCOPE)
endfunction()

function(run_legacy_stage_backend backend config_name)
    set(output_dir
        "${BINARY_DIR}/test-output/${TEST_MODE}/legacy-${backend}")
    file(REMOVE_RECURSE "${output_dir}")
    file(MAKE_DIRECTORY "${output_dir}")

    execute_process(
        COMMAND
            "${SIMULATOR}"
            --simulation_config
            "${SOURCE_DIR}/tests/fixtures/smoke/${config_name}"
            --output_path
            "${output_dir}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE simulator_result
        OUTPUT_VARIABLE simulator_stdout
        ERROR_VARIABLE simulator_stderr
        TIMEOUT 60
    )

    file(WRITE "${output_dir}/test-process.log"
        "${simulator_stdout}\n${simulator_stderr}")
    if(NOT simulator_result EQUAL 0 OR
       NOT simulator_stdout MATCHES "Finish the simulation" OR
       NOT simulator_stdout MATCHES
           "Outstanding MemoryAccess requests at simulation end: 0")
        message(FATAL_ERROR
            "${backend} legacy stage sequence failed; see "
            "${output_dir}/test-process.log")
    endif()
    if(simulator_stdout MATCHES "generated token [0-9]+/[0-9]+")
        message(FATAL_ERROR
            "${backend} default-disabled mode unexpectedly iterated tokens")
    endif()

    file(STRINGS "${output_dir}/_summary.tsv" summary_lines)
    list(LENGTH summary_lines summary_line_count)
    if(NOT summary_line_count EQUAL 4)
        message(FATAL_ERROR
            "${backend} legacy mode expected header plus three stage rows")
    endif()
    list(GET summary_lines 1 prefill_row)
    list(GET summary_lines 2 decode_row)
    list(GET summary_lines 3 npu_decode_row)
    if(NOT prefill_row MATCHES "^Prefill\t" OR
       NOT decode_row MATCHES "^Decode\t" OR
       NOT npu_decode_row MATCHES "^NPU_Decode\t")
        message(FATAL_ERROR
            "${backend} default-disabled mode did not preserve the legacy "
            "Prefill/Decode/NPU_Decode sequence")
    endif()
endfunction()

function(read_summary output_dir prefix)
    file(STRINGS "${output_dir}/_summary.tsv" summary_lines)
    list(GET summary_lines 1 summary_row)
    string(REPLACE "\t" ";" summary_fields "${summary_row}")
    list(GET summary_fields 1 total_cycles)
    list(GET summary_fields 2 pim_cycles)
    list(GET summary_fields 3 memory_bandwidth)
    set(${prefix}_TOTAL_CYCLES "${total_cycles}" PARENT_SCOPE)
    set(${prefix}_PIM_CYCLES "${pim_cycles}" PARENT_SCOPE)
    set(${prefix}_MEMORY_BANDWIDTH "${memory_bandwidth}" PARENT_SCOPE)
endfunction()

function(require_identical_file left_dir right_dir file_name description)
    foreach(directory IN ITEMS "${left_dir}" "${right_dir}")
        if(NOT EXISTS "${directory}/${file_name}")
            message(FATAL_ERROR
                "Missing ${file_name} while checking ${description}: ${directory}")
        endif()
    endforeach()
    file(SHA256 "${left_dir}/${file_name}" left_hash)
    file(SHA256 "${right_dir}/${file_name}" right_hash)
    if(NOT left_hash STREQUAL right_hash)
        message(FATAL_ERROR
            "${description} differs for ${file_name}: ${left_hash} vs ${right_hash}")
    endif()
endfunction()

function(require_pim_logical_parity ca_dir event_dir)
    require_identical_file(
        "${ca_dir}" "${event_dir}" icnt_traffic.json
        "iterative PIM logical request and byte traffic")

    set(ca_stats_file "${ca_dir}/dramsim3.json")
    set(event_stats_file "${event_dir}/eventdrivendram.json")
    foreach(stats_file IN ITEMS "${ca_stats_file}" "${event_stats_file}")
        if(NOT EXISTS "${stats_file}")
            message(FATAL_ERROR "Missing PIM backend statistics: ${stats_file}")
        endif()
    endforeach()
    file(READ "${ca_stats_file}" ca_stats)
    file(READ "${event_stats_file}" event_stats)
    string(JSON ca_channels LENGTH "${ca_stats}")
    string(JSON event_channels LENGTH "${event_stats}")
    if(NOT ca_channels EQUAL event_channels OR ca_channels LESS_EQUAL 0)
        message(FATAL_ERROR
            "CA/ED PIM statistics channel mismatch: "
            "${ca_channels} vs ${event_channels}")
    endif()

    math(EXPR last_channel "${ca_channels} - 1")
    set(total_pim_requests 0)
    foreach(channel RANGE 0 ${last_channel})
        foreach(metric IN ITEMS pheader gwrite comp readres pim)
            set(ca_key "num_${metric}_cmds")
            set(event_key "logical_${metric}_cmds")
            string(JSON ca_value GET "${ca_stats}" "${channel}" "${ca_key}")
            string(JSON event_value GET
                "${event_stats}" "${channel}" "${event_key}")
            if(NOT ca_value EQUAL event_value)
                message(FATAL_ERROR
                    "CA/ED logical PIM ${metric} mismatch on channel "
                    "${channel}: ${ca_value} vs ${event_value}")
            endif()
        endforeach()

        string(JSON ca_pim_done GET
            "${ca_stats}" "${channel}" "num_pim_cmds")
        string(JSON event_pim_requests GET
            "${event_stats}" "${channel}" "logical_pim_requests")
        string(JSON event_pim_done GET
            "${event_stats}" "${channel}" "logical_num_pim_done")
        if(NOT ca_pim_done EQUAL event_pim_requests OR
           NOT ca_pim_done EQUAL event_pim_done)
            message(FATAL_ERROR
                "CA/ED PIM request/completion mismatch on channel ${channel}: "
                "CA executed ${ca_pim_done}, ED requested/completed "
                "${event_pim_requests}/${event_pim_done}")
        endif()
        math(EXPR total_pim_requests
            "${total_pim_requests} + ${ca_pim_done}")
    endforeach()
    if(total_pim_requests LESS_EQUAL 0)
        message(FATAL_ERROR
            "Iterative Decode parity test generated no logical PIM traffic")
    endif()
    message(STATUS
        "CA/ED logical PIM parity passed for ${total_pim_requests} requests")
endfunction()

function(read_traffic output_dir prefix)
    file(READ "${output_dir}/icnt_traffic.json" traffic_json)
    string(JSON channel_count LENGTH "${traffic_json}")
    if(NOT channel_count EQUAL 2)
        message(FATAL_ERROR
            "Expected two traffic channels, found ${channel_count}")
    endif()

    set(read_bytes 0)
    set(write_bytes 0)
    set(pim_requests 0)
    math(EXPR last_channel "${channel_count} - 1")
    foreach(channel RANGE 0 ${last_channel})
        string(JSON channel_read_bytes GET
            "${traffic_json}" "${channel}" logical_read_bytes)
        string(JSON channel_write_bytes GET
            "${traffic_json}" "${channel}" logical_write_bytes)
        math(EXPR read_bytes "${read_bytes} + ${channel_read_bytes}")
        math(EXPR write_bytes "${write_bytes} + ${channel_write_bytes}")

        foreach(counter IN ITEMS
                logical_pheader_requests
                logical_gwrite_requests
                logical_comp_requests
                logical_readres_requests)
            string(JSON counter_value GET
                "${traffic_json}" "${channel}" "${counter}")
            math(EXPR pim_requests "${pim_requests} + ${counter_value}")
        endforeach()
    endforeach()

    set(${prefix}_READ_BYTES "${read_bytes}" PARENT_SCOPE)
    set(${prefix}_WRITE_BYTES "${write_bytes}" PARENT_SCOPE)
    set(${prefix}_PIM_REQUESTS "${pim_requests}" PARENT_SCOPE)
endfunction()

function(read_backend_counters stats_file prefix read_done_key write_done_key
         read_command_key write_command_key)
    file(READ "${stats_file}" stats_json)
    string(JSON channel_count LENGTH "${stats_json}")
    set(reads_done 0)
    set(writes_done 0)
    set(read_commands 0)
    set(write_commands 0)
    math(EXPR last_channel "${channel_count} - 1")
    foreach(channel RANGE 0 ${last_channel})
        string(JSON value GET "${stats_json}" "${channel}" "${read_done_key}")
        math(EXPR reads_done "${reads_done} + ${value}")
        string(JSON value GET "${stats_json}" "${channel}" "${write_done_key}")
        math(EXPR writes_done "${writes_done} + ${value}")
        string(JSON value GET "${stats_json}" "${channel}" "${read_command_key}")
        math(EXPR read_commands "${read_commands} + ${value}")
        string(JSON value GET "${stats_json}" "${channel}" "${write_command_key}")
        math(EXPR write_commands "${write_commands} + ${value}")
    endforeach()

    set(${prefix}_READS_DONE "${reads_done}" PARENT_SCOPE)
    set(${prefix}_WRITES_DONE "${writes_done}" PARENT_SCOPE)
    set(${prefix}_READ_COMMANDS "${read_commands}" PARENT_SCOPE)
    set(${prefix}_WRITE_COMMANDS "${write_commands}" PARENT_SCOPE)
endfunction()

function(require_data_container_stats output_dir expected_enabled)
    set(stats_file "${output_dir}/data_container_stats.json")
    if(NOT EXISTS "${stats_file}")
        message(FATAL_ERROR
            "Missing DataContainer statistics file: ${stats_file}")
    endif()

    file(READ "${stats_file}" stats_json)
    string(JSON enabled GET "${stats_json}" enabled)
    if(expected_enabled AND NOT enabled)
        message(FATAL_ERROR "DataContainer statistics should be enabled")
    elseif(NOT expected_enabled AND enabled)
        message(FATAL_ERROR "DataContainer statistics should be disabled")
    endif()

    foreach(key IN ITEMS
            stored_column_count peak_stored_column_count
            dram_payload_bytes pim_payload_bytes resident_payload_bytes
            peak_resident_payload_bytes payload_limit_bytes
            nonzero_payload_bytes content_fingerprint)
        string(JSON ${key} GET "${stats_json}" "${key}")
    endforeach()
    foreach(key IN ITEMS read write pheader gwrite comp readres)
        string(JSON response_${key} GET
            "${stats_json}" response_counts "${key}")
    endforeach()
    math(EXPR accounted_payload
        "${dram_payload_bytes} + ${pim_payload_bytes}")
    if(NOT resident_payload_bytes EQUAL accounted_payload)
        message(FATAL_ERROR
            "DataContainer resident bytes do not equal DRAM plus PIM payload")
    endif()
    if(peak_stored_column_count LESS stored_column_count OR
       peak_resident_payload_bytes LESS resident_payload_bytes)
        message(FATAL_ERROR
            "DataContainer peak statistics are below current occupancy")
    endif()
    if(nonzero_payload_bytes GREATER resident_payload_bytes)
        message(FATAL_ERROR
            "DataContainer nonzero byte count exceeds resident payload")
    endif()
    if(NOT expected_enabled AND
       (NOT stored_column_count EQUAL 0 OR
        NOT peak_stored_column_count EQUAL 0 OR
        NOT resident_payload_bytes EQUAL 0 OR
        NOT peak_resident_payload_bytes EQUAL 0 OR
        NOT nonzero_payload_bytes EQUAL 0 OR
        NOT content_fingerprint STREQUAL "0" OR
        NOT response_read EQUAL 0 OR
        NOT response_write EQUAL 0 OR
        NOT response_pheader EQUAL 0 OR
        NOT response_gwrite EQUAL 0 OR
        NOT response_comp EQUAL 0 OR
        NOT response_readres EQUAL 0))
        message(FATAL_ERROR
            "Disabled DataContainer unexpectedly retained payload")
    endif()
    if(expected_enabled AND resident_payload_bytes GREATER 0 AND
       content_fingerprint STREQUAL "0")
        message(FATAL_ERROR
            "Enabled DataContainer produced an empty content fingerprint")
    endif()

    string(JSON channel_count LENGTH "${stats_json}" channels)
    if(channel_count LESS_EQUAL 0)
        message(FATAL_ERROR
            "DataContainer statistics contain no channel records")
    endif()
    set(channel_pim_payload 0)
    math(EXPR last_channel "${channel_count} - 1")
    foreach(channel RANGE 0 ${last_channel})
        foreach(key IN ITEMS
                pim_input_payload_bytes peak_pim_input_payload_bytes
                pim_output_payload_bytes peak_pim_output_payload_bytes)
            string(JSON ${key} GET
                "${stats_json}" channels ${channel} "${key}")
        endforeach()
        if(peak_pim_input_payload_bytes LESS pim_input_payload_bytes OR
           peak_pim_output_payload_bytes LESS pim_output_payload_bytes)
            message(FATAL_ERROR
                "DataContainer channel ${channel} peak is below occupancy")
        endif()
        math(EXPR channel_pim_payload
            "${channel_pim_payload} + ${pim_input_payload_bytes} + ${pim_output_payload_bytes}")
    endforeach()
    if(NOT channel_pim_payload EQUAL pim_payload_bytes)
        message(FATAL_ERROR
            "DataContainer per-channel PIM bytes do not match total")
    endif()
endfunction()

function(require_smoke_workload prefix)
    if(NOT ${prefix}_READ_BYTES EQUAL 1024)
        message(FATAL_ERROR
            "Smoke workload read-byte count changed: ${${prefix}_READ_BYTES}")
    endif()
    if(NOT ${prefix}_WRITE_BYTES EQUAL 512)
        message(FATAL_ERROR
            "Smoke workload write-byte count changed: ${${prefix}_WRITE_BYTES}")
    endif()
    if(NOT ${prefix}_PIM_REQUESTS EQUAL 0)
        message(FATAL_ERROR
            "Add smoke workload unexpectedly issued PIM requests")
    endif()
    if(NOT ${prefix}_READS_DONE EQUAL 32 OR
       NOT ${prefix}_WRITES_DONE EQUAL 16)
        message(FATAL_ERROR
            "Smoke workload completion count changed: reads/writes "
            "${${prefix}_READS_DONE}/${${prefix}_WRITES_DONE}")
    endif()
    if(${prefix}_READ_COMMANDS LESS_EQUAL 0 OR
       ${prefix}_READ_COMMANDS GREATER ${prefix}_READS_DONE)
        message(FATAL_ERROR
            "Invalid read-command count: ${${prefix}_READ_COMMANDS}")
    endif()
    if(${prefix}_WRITE_COMMANDS GREATER ${prefix}_WRITES_DONE)
        message(FATAL_ERROR
            "Invalid write-command count: ${${prefix}_WRITE_COMMANDS}")
    endif()
endfunction()

if(TEST_MODE STREQUAL "full-simulation-smoke")
    run_backend(event-driven simulation_event_driven.json event_output)
    read_traffic("${event_output}" EVENT)
    read_backend_counters(
        "${event_output}/eventdrivendram.json"
        EVENT num_reads_done num_writes_done num_read_cmds num_write_cmds)
    require_smoke_workload(EVENT)
    message(STATUS
        "Full simulation smoke passed: reads/writes "
        "${EVENT_READS_DONE}/${EVENT_WRITES_DONE}, commands "
        "${EVENT_READ_COMMANDS}/${EVENT_WRITE_COMMANDS}")
elseif(TEST_MODE STREQUAL "ca-ed-consistency")
    run_backend(cycle-accurate simulation_cycle_accurate.json ca_output)
    run_backend(event-driven simulation_event_driven.json event_output)

    read_summary("${ca_output}" CA)
    read_summary("${event_output}" EVENT)
    read_traffic("${ca_output}" CA)
    read_traffic("${event_output}" EVENT)
    read_backend_counters(
        "${ca_output}/proportional_command_compensation.json"
        CA logical_num_reads_done logical_num_writes_done
        logical_num_read_cmds logical_num_write_cmds)
    read_backend_counters(
        "${event_output}/eventdrivendram.json"
        EVENT num_reads_done num_writes_done num_read_cmds num_write_cmds)

    require_smoke_workload(CA)
    require_smoke_workload(EVENT)

    foreach(metric IN ITEMS
            TOTAL_CYCLES PIM_CYCLES READ_BYTES WRITE_BYTES PIM_REQUESTS
            READS_DONE WRITES_DONE)
        if(NOT CA_${metric} EQUAL EVENT_${metric})
            message(FATAL_ERROR
                "CA/ED ${metric} mismatch: ${CA_${metric}} vs ${EVENT_${metric}}")
        endif()
    endforeach()

    if(NOT CA_READ_COMMANDS EQUAL EVENT_READ_COMMANDS)
        message(FATAL_ERROR
            "CA/ED read-command mismatch: "
            "${CA_READ_COMMANDS} vs ${EVENT_READ_COMMANDS}")
    endif()

    if(NOT CA_TOTAL_CYCLES EQUAL 205 OR NOT CA_PIM_CYCLES EQUAL 0)
        message(FATAL_ERROR
            "CA/ED Add timing baseline changed: cycles/PIM cycles "
            "${CA_TOTAL_CYCLES}/${CA_PIM_CYCLES}")
    endif()
    require_identical_file(
        "${ca_output}" "${event_output}" core_timing.tsv
        "CA/ED per-core timing and operator counters")
    require_identical_file(
        "${ca_output}" "${event_output}" icnt_traffic.json
        "CA/ED logical interconnect traffic")

    # CycleAccurate response_only mode may make writes visible to the core
    # before NewtonSim issues the corresponding physical write commands.
    # Completion and traffic counts must agree; write-command counts are
    # reported below but intentionally are not required to be identical.
    if(NOT CA_WRITE_COMMANDS EQUAL 0 OR NOT EVENT_WRITE_COMMANDS EQUAL 16)
        message(FATAL_ERROR
            "CA/ED physical write-command semantics changed: "
            "${CA_WRITE_COMMANDS} vs ${EVENT_WRITE_COMMANDS}")
    endif()

    message(STATUS
        "CA/ED consistency passed: logical reads/writes "
        "${CA_READS_DONE}/${CA_WRITES_DONE}; command counts CA "
        "${CA_READ_COMMANDS}/${CA_WRITE_COMMANDS}, ED "
        "${EVENT_READ_COMMANDS}/${EVENT_WRITE_COMMANDS}")
elseif(TEST_MODE STREQUAL "memory-access-gemm")
    run_backend(cycle-accurate simulation_gemm_cycle_accurate.json ca_output)
    run_backend(event-driven simulation_gemm_event_driven.json event_output)
    run_backend(
        cycle-accurate-data-container
        simulation_gemm_data_container_cycle_accurate.json
        dc_ca_output)
    run_backend(
        event-driven-data-container
        simulation_gemm_data_container_event_driven.json
        dc_event_output)

    read_summary("${ca_output}" CA)
    read_summary("${event_output}" EVENT)
    read_traffic("${ca_output}" CA)
    read_traffic("${event_output}" EVENT)
    read_backend_counters(
        "${ca_output}/proportional_command_compensation.json"
        CA logical_num_reads_done logical_num_writes_done
        logical_num_read_cmds logical_num_write_cmds)
    read_backend_counters(
        "${event_output}/eventdrivendram.json"
        EVENT num_reads_done num_writes_done num_read_cmds num_write_cmds)
    read_summary("${dc_ca_output}" DC_CA)
    read_summary("${dc_event_output}" DC_EVENT)
    read_traffic("${dc_ca_output}" DC_CA)
    read_traffic("${dc_event_output}" DC_EVENT)
    read_backend_counters(
        "${dc_ca_output}/proportional_command_compensation.json"
        DC_CA logical_num_reads_done logical_num_writes_done
        logical_num_read_cmds logical_num_write_cmds)
    read_backend_counters(
        "${dc_event_output}/eventdrivendram.json"
        DC_EVENT num_reads_done num_writes_done num_read_cmds num_write_cmds)
    require_data_container_stats("${ca_output}" FALSE)
    require_data_container_stats("${event_output}" FALSE)
    require_data_container_stats("${dc_ca_output}" TRUE)
    require_data_container_stats("${dc_event_output}" TRUE)

    foreach(prefix IN ITEMS CA EVENT DC_CA DC_EVENT)
        if(NOT ${prefix}_TOTAL_CYCLES EQUAL 869 OR
           NOT ${prefix}_READ_BYTES EQUAL 8832 OR
           NOT ${prefix}_WRITE_BYTES EQUAL 2048 OR
           NOT ${prefix}_PIM_REQUESTS EQUAL 0 OR
           NOT ${prefix}_READS_DONE EQUAL 276 OR
           NOT ${prefix}_WRITES_DONE EQUAL 64 OR
           NOT ${prefix}_READ_COMMANDS EQUAL 268)
            message(FATAL_ERROR
                "${prefix} GEMM ownership regression: cycles/read-bytes/"
                "write-bytes/reads/writes/read-commands = "
                "${${prefix}_TOTAL_CYCLES}/${${prefix}_READ_BYTES}/"
                "${${prefix}_WRITE_BYTES}/${${prefix}_READS_DONE}/"
                "${${prefix}_WRITES_DONE}/${${prefix}_READ_COMMANDS}")
        endif()
    endforeach()
    if(NOT CA_WRITE_COMMANDS EQUAL 50 OR
       NOT EVENT_WRITE_COMMANDS EQUAL 64 OR
       NOT DC_CA_WRITE_COMMANDS EQUAL 50 OR
       NOT DC_EVENT_WRITE_COMMANDS EQUAL 64)
        message(FATAL_ERROR
            "GEMM write-command regression: disabled CA/ED and enabled CA/ED "
            "${CA_WRITE_COMMANDS}/${EVENT_WRITE_COMMANDS} and "
            "${DC_CA_WRITE_COMMANDS}/${DC_EVENT_WRITE_COMMANDS}")
    endif()

    require_identical_file(
        "${ca_output}" "${event_output}" core_timing.tsv
        "CA/ED GEMM per-core timing and operator counters")
    require_identical_file(
        "${ca_output}" "${event_output}" icnt_traffic.json
        "CA/ED GEMM logical interconnect traffic")
    require_identical_file(
        "${ca_output}" "${dc_ca_output}" core_timing.tsv
        "DataContainer-disabled/enabled CycleAccurate GEMM timing")
    require_identical_file(
        "${event_output}" "${dc_event_output}" core_timing.tsv
        "DataContainer-disabled/enabled EventDriven GEMM timing")
    require_identical_file(
        "${ca_output}" "${dc_ca_output}" icnt_traffic.json
        "DataContainer-disabled/enabled CycleAccurate GEMM traffic")
    require_identical_file(
        "${event_output}" "${dc_event_output}" icnt_traffic.json
        "DataContainer-disabled/enabled EventDriven GEMM traffic")
    require_identical_file(
        "${dc_ca_output}" "${dc_event_output}" data_container_stats.json
        "CycleAccurate/EventDriven DataContainer occupancy statistics")

    message(STATUS
        "MemoryAccess GEMM lifecycle passed: cycles ${CA_TOTAL_CYCLES}, "
        "logical reads/writes ${CA_READS_DONE}/${CA_WRITES_DONE}, "
        "commands CA ${CA_READ_COMMANDS}/${CA_WRITE_COMMANDS}, "
        "ED ${EVENT_READ_COMMANDS}/${EVENT_WRITE_COMMANDS}; "
        "DataContainer enabled CA ${DC_CA_READ_COMMANDS}/${DC_CA_WRITE_COMMANDS}, "
        "ED ${DC_EVENT_READ_COMMANDS}/${DC_EVENT_WRITE_COMMANDS}")
elseif(TEST_MODE STREQUAL "booksim2-smoke")
    run_backend(simple simulation_event_driven.json simple_output)
    run_backend(booksim2 simulation_booksim2_event_driven.json booksim_output)

    require_identical_file(
        "${simple_output}" "${booksim_output}" icnt_traffic.json
        "Simple/BookSim logical interconnect traffic")
    set(booksim_stats_file "${booksim_output}/booksim2_stats.json")
    if(NOT EXISTS "${booksim_stats_file}")
        message(FATAL_ERROR "BookSim run did not produce booksim2_stats.json")
    endif()
    file(READ "${booksim_stats_file}" booksim_stats)
    foreach(key IN ITEMS
            nodes flit_size_bytes booksim_cycles injected_packets
            ejected_packets injected_payload_bytes ejected_payload_bytes)
        string(JSON ${key} GET "${booksim_stats}" "${key}")
    endforeach()
    if(NOT nodes EQUAL 4 OR NOT flit_size_bytes EQUAL 16)
        message(FATAL_ERROR
            "Unexpected BookSim topology/flit geometry: "
            "${nodes}/${flit_size_bytes}")
    endif()
    if(booksim_cycles LESS_EQUAL 0 OR injected_packets LESS_EQUAL 0 OR
       NOT injected_packets EQUAL ejected_packets OR
       NOT injected_payload_bytes EQUAL ejected_payload_bytes)
        message(FATAL_ERROR
            "BookSim packet accounting is incomplete: packets "
            "${injected_packets}/${ejected_packets}, bytes "
            "${injected_payload_bytes}/${ejected_payload_bytes}")
    endif()
    file(READ "${booksim_output}/test-process.log" booksim_log)
    if(NOT booksim_log MATCHES "BookSim active" OR
       NOT booksim_log MATCHES "BookSim packets injected/ejected")
        message(FATAL_ERROR
            "BookSim run did not report backend activation and packet flow")
    endif()
    message(STATUS
        "BookSim full simulation passed: ${injected_packets} packets, "
        "${injected_payload_bytes} payload bytes, ${booksim_cycles} cycles")
elseif(TEST_MODE STREQUAL "iterative-decode")
    run_legacy_stage_backend(
        cycle-accurate simulation_legacy_stage_sequence_cycle_accurate.json)
    run_legacy_stage_backend(
        event-driven simulation_legacy_stage_sequence_event_driven.json)
    run_iterative_decode_backend(
        cycle-accurate simulation_iterative_decode_cycle_accurate.json
        iterative_ca_output)
    run_iterative_decode_backend(
        event-driven simulation_iterative_decode_event_driven.json
        iterative_event_output)
    require_pim_logical_parity(
        "${iterative_ca_output}" "${iterative_event_output}")
    message(STATUS
        "Legacy and iterative Decode passed in CycleAccurate and "
        "EventDriven modes")
elseif(TEST_MODE STREQUAL "continuous-batching")
    run_continuous_batch_backend(
        cycle-accurate simulation_continuous_batching_cycle_accurate.json
        continuous_ca_output)
    run_continuous_batch_backend(
        event-driven simulation_continuous_batching_event_driven.json
        continuous_ed_output)
    require_identical_file(
        "${continuous_ca_output}" "${continuous_ed_output}"
        icnt_traffic.json "continuous-batching interconnect traffic")
    message(STATUS
        "Continuous batching passed in CycleAccurate and EventDriven modes")
else()
    message(FATAL_ERROR "Unsupported TEST_MODE: ${TEST_MODE}")
endif()
