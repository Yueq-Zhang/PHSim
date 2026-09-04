cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(generated_root "${BINARY_DIR}/test-generated/legacy-acceleration")
set(output_root "${BINARY_DIR}/test-output/legacy-acceleration")
file(REMOVE_RECURSE "${generated_root}" "${output_root}")
file(MAKE_DIRECTORY "${generated_root}" "${output_root}")

file(READ "${SOURCE_DIR}/tests/fixtures/smoke/model_config.json" model_json)
string(JSON model_json SET "${model_json}" model_n_embd 1024)
file(WRITE "${generated_root}/model.json" "${model_json}\n")

function(make_case_config method backend output_config)
    file(READ
        "${SOURCE_DIR}/tests/fixtures/smoke/inference_gemm_event_driven.json"
        inference_json)
    string(JSON inference_json SET "${inference_json}" accelerate_ctrl true)
    string(JSON inference_json SET "${inference_json}"
        accelerate_method "\"${method}\"")
    string(JSON inference_json SET "${inference_json}"
        gen_request_input_size 512)
    if(backend STREQUAL "ed")
        string(JSON inference_json SET "${inference_json}"
            dram_trace_simulation_mode true)
    elseif(backend STREQUAL "ca")
        string(JSON inference_json SET "${inference_json}"
            dram_trace_simulation_mode false)
    else()
        message(FATAL_ERROR "Unsupported backend '${backend}'")
    endif()

    set(inference_path "${generated_root}/${method}-${backend}-inference.json")
    file(WRITE "${inference_path}" "${inference_json}\n")

    file(READ
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_gemm_event_driven.json"
        simulation_json)
    string(JSON simulation_json SET "${simulation_json}"
        inference_config_file_path "\"${inference_path}\"")
    string(JSON simulation_json SET "${simulation_json}"
        model_config_file_path "\"${generated_root}/model.json\"")
    set(simulation_path "${generated_root}/${method}-${backend}-simulation.json")
    file(WRITE "${simulation_path}" "${simulation_json}\n")
    set(${output_config} "${simulation_path}" PARENT_SCOPE)
endfunction()

function(run_acceleration_case method backend)
    make_case_config("${method}" "${backend}" simulation_config)
    set(case_output "${output_root}/${method}-${backend}")
    file(MAKE_DIRECTORY "${case_output}")
    execute_process(
        COMMAND "${SIMULATOR}"
            --simulation_config "${simulation_config}"
            --output_path "${case_output}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE simulator_result
        OUTPUT_VARIABLE simulator_stdout
        ERROR_VARIABLE simulator_stderr
        TIMEOUT 60)
    file(WRITE "${case_output}/test-process.log"
        "${simulator_stdout}\n${simulator_stderr}")

    if(NOT simulator_result EQUAL 0)
        message(FATAL_ERROR
            "${method}/${backend} failed with ${simulator_result}; see "
            "${case_output}/test-process.log")
    endif()
    if(method STREQUAL "naive")
        if(NOT simulator_stdout MATCHES "Applying naive acceleration")
            message(FATAL_ERROR
                "${method}/${backend} did not exercise its accelerated time jump")
        endif()
    elseif(NOT simulator_stdout MATCHES
           "(Applying Loop_wise acceleration|Loop_wise prediction)")
        message(FATAL_ERROR
            "${method}/${backend} did not exercise the Loop_wise prediction path")
    endif()
    if(NOT simulator_stdout MATCHES "Finish the simulation" OR
       NOT simulator_stdout MATCHES
           "Outstanding MemoryAccess requests at simulation end: 0")
        message(FATAL_ERROR
            "${method}/${backend} did not finish with a drained request graph")
    endif()
endfunction()

foreach(method IN ITEMS naive Loop_wise)
    foreach(backend IN ITEMS ca ed)
        run_acceleration_case("${method}" "${backend}")
    endforeach()
endforeach()

make_case_config("invalid-method" "ed" invalid_config)
execute_process(
    COMMAND "${SIMULATOR}"
        --simulation_config "${invalid_config}"
        --output_path "${output_root}/invalid-method"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE invalid_result
    OUTPUT_VARIABLE invalid_stdout
    ERROR_VARIABLE invalid_stderr
    TIMEOUT 20)
if(invalid_result EQUAL 0)
    message(FATAL_ERROR "Unsupported accelerate_method was accepted")
endif()
set(invalid_output "${invalid_stdout}\n${invalid_stderr}")
if(NOT invalid_output MATCHES
   "supported methods are 'naive', 'Loop_wise', and 'Proportional'")
    message(FATAL_ERROR
        "Unsupported accelerate_method did not report the allowed values")
endif()
