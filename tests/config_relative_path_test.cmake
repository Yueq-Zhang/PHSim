cmake_minimum_required(VERSION 3.22)

foreach(required_variable IN ITEMS SIMULATOR SOURCE_DIR BINARY_DIR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

set(test_root "${BINARY_DIR}/test-output/config-relative-path")
set(work_dir "${test_root}/unrelated-working-directory")
set(output_dir "${test_root}/output")
file(REMOVE_RECURSE "${test_root}")
file(MAKE_DIRECTORY "${work_dir}" "${output_dir}")

execute_process(
    COMMAND
        "${SIMULATOR}"
        --simulation_config
        "${SOURCE_DIR}/tests/fixtures/smoke/simulation_gemm_cycle_accurate.json"
        --output_path
        "${output_dir}"
    WORKING_DIRECTORY "${work_dir}"
    RESULT_VARIABLE simulator_result
    OUTPUT_VARIABLE simulator_stdout
    ERROR_VARIABLE simulator_stderr
    TIMEOUT 30
)

file(WRITE "${test_root}/test-process.log"
    "${simulator_stdout}\n${simulator_stderr}")

if(NOT simulator_result EQUAL 0)
    message(FATAL_ERROR
        "Configuration-relative launch failed with exit code "
        "${simulator_result}. See ${test_root}/test-process.log")
endif()
if(NOT simulator_stdout MATCHES "Finish the simulation")
    message(FATAL_ERROR
        "Configuration-relative launch did not complete normally")
endif()
if(simulator_stdout MATCHES "deprecated working-directory-relative")
    message(FATAL_ERROR
        "Configuration-relative fixture unexpectedly used the legacy fallback")
endif()

foreach(required_file IN ITEMS _summary.tsv icnt_traffic.json log.txt)
    if(NOT EXISTS "${output_dir}/${required_file}")
        message(FATAL_ERROR
            "Configuration-relative launch did not produce ${required_file}")
    endif()
endforeach()
