cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR TEST_SUITE)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(suite_root "${BINARY_DIR}/test-output/pruning/${TEST_SUITE}")
set(generated_root "${BINARY_DIR}/test-generated/pruning/${TEST_SUITE}")
file(REMOVE_RECURSE "${suite_root}" "${generated_root}")
file(MAKE_DIRECTORY "${suite_root}" "${generated_root}")
set(result_file "${suite_root}/results.tsv")
file(WRITE "${result_file}"
    "case\tkind\tbackend\tpruning\tvm\tdc\ttotal_cycles\tstage_cycles\tlogical_tiles\tmeasured_tiles\testimated_tiles\tlogical_calculations\testimated_calculations\tread_bytes\twrite_bytes\tpim_requests\tmapped_pages\tmapping_calls\tchanged_mappings\tmapping_pair_xor\tmapping_pair_sum\tdc_resident_bytes\tdc_nonzero_bytes\tdc_fingerprint\tdc_read_responses\tdc_write_responses\tdc_pheader_responses\tdc_gwrite_responses\tdc_comp_responses\tdc_readres_responses\n")

file(READ
    "${SOURCE_DIR}/tests/fixtures/smoke/pim_config_iterative.json"
    smoke_pim_json)
string(JSON smoke_burst_bytes GET "${smoke_pim_json}" dram_req_size)

function(json_bool output value)
    if(value)
        set(${output} true PARENT_SCOPE)
    else()
        set(${output} false PARENT_SCOPE)
    endif()
endfunction()

function(require_generated_tokens simulator_stdout request_id expected_tokens case_name)
    string(REGEX MATCHALL
        "Scheduler:: Request ${request_id} generated token [0-9]+/${expected_tokens}"
        token_progress "${simulator_stdout}")
    list(LENGTH token_progress token_count)
    if(NOT token_count EQUAL expected_tokens)
        message(FATAL_ERROR
            "${case_name}: request ${request_id} generated "
            "${token_count}/${expected_tokens} tokens")
    endif()
endfunction()

function(require_vm_replay baseline_case pruned_case context)
    set(baseline_file
        "${suite_root}/${baseline_case}/virtual_memory_stats.json")
    set(pruned_file
        "${suite_root}/${pruned_case}/virtual_memory_stats.json")
    if(NOT EXISTS "${baseline_file}" OR NOT EXISTS "${pruned_file}")
        message(FATAL_ERROR
            "${context}: missing virtual_memory_stats.json")
    endif()
    file(SHA256 "${baseline_file}" baseline_hash)
    file(SHA256 "${pruned_file}" pruned_hash)
    if(NOT baseline_hash STREQUAL pruned_hash)
        message(FATAL_ERROR
            "${context}: pruned virtual-memory mapping state differs from "
            "baseline ${baseline_case}")
    endif()
endfunction()

function(run_case case_name kind backend pruning vm dc)
    if(kind STREQUAL "tile")
        set(base_inference
            "${SOURCE_DIR}/tests/fixtures/smoke/inference_gemm_event_driven.json")
    elseif(kind STREQUAL "fixed")
        set(base_inference
            "${SOURCE_DIR}/tests/fixtures/smoke/inference_iterative_decode_event_driven.json")
    elseif(kind STREQUAL "continuous")
        set(base_inference
            "${SOURCE_DIR}/tests/fixtures/smoke/inference_continuous_batching_event_driven.json")
    else()
        message(FATAL_ERROR "Unsupported pruning test kind: ${kind}")
    endif()

    file(READ "${base_inference}" inference_json)
    json_bool(vm_json ${vm})
    json_bool(dc_json ${dc})
    if(backend STREQUAL "ed")
        set(event_json true)
    elseif(backend STREQUAL "ca")
        set(event_json false)
    else()
        message(FATAL_ERROR "Unsupported DRAM backend: ${backend}")
    endif()
    string(JSON inference_json SET "${inference_json}"
        virtual_mem_hash_enable ${vm_json})
    string(JSON inference_json SET "${inference_json}"
        dram_data_container_enable ${dc_json})
    string(JSON inference_json SET "${inference_json}"
        dram_trace_simulation_mode ${event_json})
    if(dc)
        string(JSON inference_json SET "${inference_json}"
            dram_data_container_max_payload_mb 16)
    else()
        string(JSON inference_json SET "${inference_json}"
            dram_data_container_max_payload_mb 0)
    endif()

    string(JSON inference_json SET "${inference_json}"
        accelerate_ctrl false)
    string(JSON inference_json SET "${inference_json}"
        accelerate_method "\"Proportional\"")
    string(JSON inference_json SET "${inference_json}"
        accelerate_sample_ratio 0.5)
    string(JSON inference_json SET "${inference_json}"
        compile_time_tile_pruning false)
    string(JSON inference_json SET "${inference_json}"
        decode_pruning_enabled false)
    string(JSON inference_json SET "${inference_json}"
        decode_pruning_iterations 3)
    string(JSON inference_json SET "${inference_json}"
        decode_pruning_sample_iterations 1)
    if(pruning STREQUAL "proportional-runtime")
        string(JSON inference_json SET "${inference_json}"
            accelerate_ctrl true)
    elseif(pruning STREQUAL "proportional-compile")
        string(JSON inference_json SET "${inference_json}"
            accelerate_ctrl true)
        string(JSON inference_json SET "${inference_json}"
            compile_time_tile_pruning true)
    elseif(pruning STREQUAL "decode")
        string(JSON inference_json SET "${inference_json}"
            decode_pruning_enabled true)
    elseif(NOT pruning STREQUAL "baseline")
        message(FATAL_ERROR "Unsupported pruning mode: ${pruning}")
    endif()

    if(kind STREQUAL "tile")
        string(JSON inference_json SET "${inference_json}"
            max_batch_size 1)
        string(JSON inference_json SET "${inference_json}"
            max_active_reqs 1)
        string(JSON inference_json SET "${inference_json}"
            batch_scheduler "\"legacy\"")
        string(JSON inference_json SET "${inference_json}"
            test_single_op true)
        string(JSON inference_json SET "${inference_json}"
            test_single_op_name "\"gemm\"")
        string(JSON inference_json SET "${inference_json}"
            gen_request true)
        string(JSON inference_json SET "${inference_json}"
            gen_request_count 1)
        string(JSON inference_json SET "${inference_json}"
            gen_request_input_size 4)
        string(JSON inference_json SET "${inference_json}"
            gen_request_output_size 0)
        string(JSON inference_json SET "${inference_json}"
            output_token_iteration_enable false)
    elseif(kind STREQUAL "fixed")
        string(JSON inference_json SET "${inference_json}"
            max_batch_size 2)
        string(JSON inference_json SET "${inference_json}"
            max_active_reqs 2)
        string(JSON inference_json SET "${inference_json}"
            batch_scheduler "\"legacy\"")
        string(JSON inference_json SET "${inference_json}"
            gen_request true)
        string(JSON inference_json SET "${inference_json}"
            gen_request_count 2)
        string(JSON inference_json SET "${inference_json}"
            gen_request_input_size 4)
        string(JSON inference_json SET "${inference_json}"
            gen_request_output_size 3)
        string(JSON inference_json SET "${inference_json}"
            output_token_iteration_enable true)
    endif()

    set(case_generated_dir "${generated_root}/${case_name}")
    set(case_output_dir "${suite_root}/${case_name}")
    file(MAKE_DIRECTORY "${case_generated_dir}" "${case_output_dir}")
    set(inference_path "${case_generated_dir}/inference.json")
    file(WRITE "${inference_path}" "${inference_json}\n")

    file(READ
        "${SOURCE_DIR}/tests/fixtures/smoke/compute_config.json"
        compute_json)
    if(kind STREQUAL "tile")
        # Force at least two complete GEMM K-loops on the one-core smoke
        # machine; the regular 1 MiB scratchpad fits the whole 256x256 GEMM
        # in one tile and therefore cannot exercise Tile Pruning.
        string(JSON compute_json SET "${compute_json}" spad_size 128)
        string(JSON compute_json SET "${compute_json}" accum_spad_size 128)
    endif()
    if(vm)
        # Force core-to-ICNT backpressure.  A request must be mapped only when
        # this one-entry queue accepts it, not once per stalled simulator tick.
        string(JSON compute_json SET "${compute_json}"
            icnt_input_buffer_size 1)
        if(kind STREQUAL "continuous")
            # Exercise mapping-only Decode replay across more than one core;
            # single-core ordering alone cannot validate per-core replay.
            string(JSON compute_json SET "${compute_json}" num_cores 2)
        endif()
    endif()
    set(compute_path "${case_generated_dir}/compute.json")
    file(WRITE "${compute_path}" "${compute_json}\n")

    foreach(path_variable IN ITEMS
            inference_path compute_path case_generated_dir case_output_dir
            SOURCE_DIR)
        file(TO_CMAKE_PATH "${${path_variable}}" ${path_variable}_json)
    endforeach()
    set(simulation_json "{}")
    string(JSON simulation_json SET "${simulation_json}"
        compute_die_config_file_path
        "\"${compute_path_json}\"")
    string(JSON simulation_json SET "${simulation_json}"
        DRAM_config_file_path
        "\"${SOURCE_DIR_json}/tests/fixtures/smoke/memory_config.ini\"")
    string(JSON simulation_json SET "${simulation_json}"
        PIM_config_file_path
        "\"${SOURCE_DIR_json}/tests/fixtures/smoke/pim_config_iterative.json\"")
    string(JSON simulation_json SET "${simulation_json}"
        model_config_file_path
        "\"${SOURCE_DIR_json}/tests/fixtures/smoke/model_config_iterative_pim.json\"")
    string(JSON simulation_json SET "${simulation_json}"
        inference_config_file_path "\"${inference_path_json}\"")
    string(JSON simulation_json SET "${simulation_json}"
        request_file_path
        "\"${SOURCE_DIR_json}/tests/fixtures/smoke/request_trace_continuous.csv\"")
    set(simulation_path "${case_generated_dir}/simulation.json")
    file(WRITE "${simulation_path}" "${simulation_json}\n")

    execute_process(
        COMMAND "${SIMULATOR}"
            --simulation_config "${simulation_path}"
            --output_path "${case_output_dir}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE simulator_result
        OUTPUT_VARIABLE simulator_stdout
        ERROR_VARIABLE simulator_stderr
        TIMEOUT 90)
    file(WRITE "${case_output_dir}/test-process.log"
        "${simulator_stdout}\n${simulator_stderr}")

    if(dc AND NOT pruning STREQUAL "baseline")
        if(simulator_result EQUAL 0)
            message(FATAL_ERROR
                "${case_name}: incompatible Pruning+DataContainer config "
                "was accepted")
        endif()
        if(NOT "${simulator_stdout}\n${simulator_stderr}" MATCHES
               "dram_data_container_enable=true is incompatible")
            message(FATAL_ERROR
                "${case_name}: incompatible configuration did not report "
                "the DataContainer conflict")
        endif()
        message(STATUS
            "${case_name}: incompatible Pruning+DataContainer config rejected")
        return()
    endif()

    if(NOT simulator_result EQUAL 0 OR
       NOT simulator_stdout MATCHES "Finish the simulation" OR
       NOT simulator_stdout MATCHES
           "Outstanding MemoryAccess requests at simulation end: 0")
        message(FATAL_ERROR
            "${case_name} did not finish cleanly (exit ${simulator_result}); "
            "see ${case_output_dir}/test-process.log")
    endif()

    if(kind STREQUAL "fixed")
        require_generated_tokens("${simulator_stdout}" 0 3 "${case_name}")
        require_generated_tokens("${simulator_stdout}" 1 3 "${case_name}")
        set(expected_responses 2)
    elseif(kind STREQUAL "continuous")
        require_generated_tokens("${simulator_stdout}" 0 3 "${case_name}")
        require_generated_tokens("${simulator_stdout}" 1 1 "${case_name}")
        require_generated_tokens("${simulator_stdout}" 2 2 "${case_name}")
        foreach(expected_batch IN ITEMS
                "stage=Prefill, requests=[0,1]"
                "stage=Decode, requests=[0,1]"
                "stage=Prefill, requests=[2]"
                "stage=Decode, requests=[0,2]")
            string(FIND "${simulator_stdout}" "${expected_batch}"
                batch_position)
            if(batch_position EQUAL -1)
                message(FATAL_ERROR
                    "${case_name} missed continuous batch ${expected_batch}")
            endif()
        endforeach()
        set(expected_responses 3)
    endif()
    if(DEFINED expected_responses)
        string(REGEX MATCHALL "Client Receive response From Scheduler!"
            client_responses "${simulator_stdout}")
        list(LENGTH client_responses response_count)
        if(NOT response_count EQUAL expected_responses)
            message(FATAL_ERROR
                "${case_name} completed ${response_count}/${expected_responses} requests")
        endif()
        unset(expected_responses)
    endif()

    foreach(required_file IN ITEMS
            _summary.tsv icnt_traffic.json data_container_stats.json
            virtual_memory_stats.json)
        if(NOT EXISTS "${case_output_dir}/${required_file}")
            message(FATAL_ERROR
                "${case_name} did not produce ${required_file}")
        endif()
    endforeach()

    file(STRINGS "${case_output_dir}/_summary.tsv" summary_lines)
    list(REMOVE_AT summary_lines 0)
    set(total_cycles 0)
    set(stage_cycles "")
    foreach(summary_row IN LISTS summary_lines)
        string(REPLACE "\t" ";" summary_fields "${summary_row}")
        list(GET summary_fields 0 stage_name)
        list(GET summary_fields 1 stage_cycle)
        math(EXPR total_cycles "${total_cycles} + ${stage_cycle}")
        if(stage_cycles STREQUAL "")
            set(stage_cycles "${stage_name}:${stage_cycle}")
        else()
            string(APPEND stage_cycles "|${stage_name}:${stage_cycle}")
        endif()
    endforeach()

    file(READ "${case_output_dir}/icnt_traffic.json" traffic_json)
    string(JSON channel_count LENGTH "${traffic_json}")
    math(EXPR last_channel "${channel_count} - 1")
    set(read_bytes 0)
    set(write_bytes 0)
    set(pim_requests 0)
    foreach(channel RANGE 0 ${last_channel})
        string(JSON value GET "${traffic_json}" ${channel}
            logical_read_bytes)
        math(EXPR read_bytes "${read_bytes} + ${value}")
        string(JSON value GET "${traffic_json}" ${channel}
            logical_write_bytes)
        math(EXPR write_bytes "${write_bytes} + ${value}")
        foreach(counter IN ITEMS
                logical_pheader_requests logical_gwrite_requests
                logical_comp_requests logical_readres_requests)
            string(JSON value GET "${traffic_json}" ${channel} ${counter})
            math(EXPR pim_requests "${pim_requests} + ${value}")
        endforeach()
    endforeach()

    set(logical_tiles 0)
    set(measured_tiles 0)
    set(estimated_tiles 0)
    set(logical_calculations 0)
    set(estimated_calculations 0)
    string(REGEX MATCHALL "Operation workload [^\r\n]*"
        workload_lines "${simulator_stdout}")
    foreach(workload_line IN LISTS workload_lines)
        string(REGEX MATCH
            "Logical tiles ([0-9]+), Measured tiles ([0-9]+), Estimated tiles ([0-9]+), Memory reads ([0-9]+), Memory writes ([0-9]+), Estimated memory reads ([0-9]+), Estimated memory writes ([0-9]+), Calculations ([0-9]+), Estimated calculations ([0-9]+)"
            workload_match "${workload_line}")
        if(workload_match)
            math(EXPR logical_tiles "${logical_tiles} + ${CMAKE_MATCH_1}")
            math(EXPR measured_tiles "${measured_tiles} + ${CMAKE_MATCH_2}")
            math(EXPR estimated_tiles "${estimated_tiles} + ${CMAKE_MATCH_3}")
            math(EXPR logical_calculations
                "${logical_calculations} + ${CMAKE_MATCH_8}")
            math(EXPR estimated_calculations
                "${estimated_calculations} + ${CMAKE_MATCH_9}")
        endif()
    endforeach()

    if(NOT dc AND NOT pruning STREQUAL "baseline" AND
           estimated_tiles EQUAL 0)
        message(FATAL_ERROR
            "${case_name}: Pruning did not remain active without DataContainer")
    endif()

    file(READ "${case_output_dir}/virtual_memory_stats.json" vm_json)
    foreach(key IN ITEMS mapped_page_count mapping_call_count
            changed_mapping_count mapping_pair_xor mapping_pair_sum)
        string(JSON vm_${key} GET "${vm_json}" ${key})
    endforeach()
    if(vm)
        math(EXPR read_remainder "${read_bytes} % ${smoke_burst_bytes}")
        math(EXPR write_remainder "${write_bytes} % ${smoke_burst_bytes}")
        if(NOT read_remainder EQUAL 0 OR NOT write_remainder EQUAL 0)
            message(FATAL_ERROR
                "${case_name}: logical byte traffic is not burst aligned")
        endif()
        math(EXPR expected_mapping_calls
            "${read_bytes} / ${smoke_burst_bytes} + ${write_bytes} / ${smoke_burst_bytes} + ${pim_requests}")
        if(NOT vm_mapping_call_count EQUAL expected_mapping_calls)
            message(FATAL_ERROR
                "${case_name}: virtual mapping ran ${vm_mapping_call_count} "
                "times for ${expected_mapping_calls} actual requests; "
                "a back-pressured request may have been remapped")
        endif()
    endif()

    file(READ "${case_output_dir}/data_container_stats.json" dc_stats_json)
    foreach(key IN ITEMS resident_payload_bytes nonzero_payload_bytes
            content_fingerprint)
        string(JSON dc_${key} GET "${dc_stats_json}" ${key})
    endforeach()
    foreach(key IN ITEMS read write pheader gwrite comp readres)
        string(JSON dc_${key}_responses GET
            "${dc_stats_json}" response_counts ${key})
    endforeach()

    file(APPEND "${result_file}"
        "${case_name}\t${kind}\t${backend}\t${pruning}\t${vm}\t${dc}\t${total_cycles}\t${stage_cycles}\t${logical_tiles}\t${measured_tiles}\t${estimated_tiles}\t${logical_calculations}\t${estimated_calculations}\t${read_bytes}\t${write_bytes}\t${pim_requests}\t${vm_mapped_page_count}\t${vm_mapping_call_count}\t${vm_changed_mapping_count}\t${vm_mapping_pair_xor}\t${vm_mapping_pair_sum}\t${dc_resident_payload_bytes}\t${dc_nonzero_payload_bytes}\t${dc_content_fingerprint}\t${dc_read_responses}\t${dc_write_responses}\t${dc_pheader_responses}\t${dc_gwrite_responses}\t${dc_comp_responses}\t${dc_readres_responses}\n")
    message(STATUS
        "${case_name}: cycles=${total_cycles}, logical tiles="
        "${logical_tiles}, estimated=${estimated_tiles}, traffic="
        "${read_bytes}/${write_bytes}/${pim_requests}, VM calls="
        "${vm_mapping_call_count}, DC responses="
        "${dc_read_responses}/${dc_write_responses}/"
        "${dc_pheader_responses}/${dc_gwrite_responses}/"
        "${dc_comp_responses}/${dc_readres_responses}")
endfunction()

if(TEST_SUITE STREQUAL "tile")
    foreach(backend IN ITEMS ca ed)
        run_case("none-baseline-${backend}" tile ${backend}
            baseline false false)
        run_case("none-runtime-${backend}" tile ${backend}
            proportional-runtime false false)
        run_case("none-compile-${backend}" tile ${backend}
            proportional-compile false false)
        run_case("vm-baseline-${backend}" tile ${backend}
            baseline true false)
        run_case("vm-compile-${backend}" tile ${backend}
            proportional-compile true false)
        run_case("dc-baseline-${backend}" tile ${backend}
            baseline false true)
        run_case("dc-compile-${backend}" tile ${backend}
            proportional-compile false true)
        require_vm_replay(
            "vm-baseline-${backend}" "vm-compile-${backend}"
            "Tile Pruning virtual-memory replay (${backend})")
    endforeach()
elseif(TEST_SUITE STREQUAL "decode-fixed")
    foreach(backend IN ITEMS ca ed)
        run_case("baseline-${backend}" fixed ${backend}
            baseline false false)
        run_case("pruned-${backend}" fixed ${backend}
            decode false false)
    endforeach()
elseif(TEST_SUITE STREQUAL "decode-continuous")
    foreach(feature IN ITEMS none vm dc)
        if(feature STREQUAL "vm")
            set(vm true)
            set(dc false)
        elseif(feature STREQUAL "dc")
            set(vm false)
            set(dc true)
        else()
            set(vm false)
            set(dc false)
        endif()
        foreach(backend IN ITEMS ca ed)
            run_case("${feature}-baseline-${backend}" continuous ${backend}
                baseline ${vm} ${dc})
            run_case("${feature}-pruned-${backend}" continuous ${backend}
                decode ${vm} ${dc})
            if(feature STREQUAL "vm")
                require_vm_replay(
                    "${feature}-baseline-${backend}"
                    "${feature}-pruned-${backend}"
                    "Decode Pruning virtual-memory replay (${backend})")
            endif()
        endforeach()
    endforeach()
elseif(TEST_SUITE STREQUAL "triple")
    foreach(backend IN ITEMS ca ed)
        run_case("tile-baseline-${backend}" tile ${backend}
            baseline true true)
        run_case("tile-pruned-${backend}" tile ${backend}
            proportional-compile true true)
        run_case("decode-baseline-${backend}" continuous ${backend}
            baseline true true)
        run_case("decode-pruned-${backend}" continuous ${backend}
            decode true true)
    endforeach()
else()
    message(FATAL_ERROR "Unsupported pruning TEST_SUITE: ${TEST_SUITE}")
endif()

message(STATUS "Pruning composition suite ${TEST_SUITE} wrote ${result_file}")
