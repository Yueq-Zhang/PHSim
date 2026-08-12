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

function(read_summary output_dir prefix)
    file(STRINGS "${output_dir}/_summary.tsv" summary_lines)
    list(GET summary_lines 1 summary_row)
    string(REPLACE "\t" ";" summary_fields "${summary_row}")
    list(GET summary_fields 1 total_cycles)
    list(GET summary_fields 3 memory_bandwidth)
    set(${prefix}_TOTAL_CYCLES "${total_cycles}" PARENT_SCOPE)
    set(${prefix}_MEMORY_BANDWIDTH "${memory_bandwidth}" PARENT_SCOPE)
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
            READ_BYTES WRITE_BYTES PIM_REQUESTS READS_DONE WRITES_DONE)
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

    # CycleAccurate response_only mode may make writes visible to the core
    # before NewtonSim issues the corresponding physical write commands.
    # Completion and traffic counts must agree; write-command counts are
    # reported below but intentionally are not required to be identical.

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

    message(STATUS
        "MemoryAccess GEMM lifecycle passed: cycles ${CA_TOTAL_CYCLES}, "
        "logical reads/writes ${CA_READS_DONE}/${CA_WRITES_DONE}, "
        "commands CA ${CA_READ_COMMANDS}/${CA_WRITE_COMMANDS}, "
        "ED ${EVENT_READ_COMMANDS}/${EVENT_WRITE_COMMANDS}; "
        "DataContainer enabled CA ${DC_CA_READ_COMMANDS}/${DC_CA_WRITE_COMMANDS}, "
        "ED ${DC_EVENT_READ_COMMANDS}/${DC_EVENT_WRITE_COMMANDS}")
else()
    message(FATAL_ERROR "Unsupported TEST_MODE: ${TEST_MODE}")
endif()
