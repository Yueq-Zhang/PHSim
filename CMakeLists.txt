cmake_minimum_required(VERSION 3.22)
project(NMC_Simulator)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 路径设置
set(INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)
include_directories(${INCLUDE_PATH})

# 查找 protobuf 库
#find_package(Protobuf REQUIRED)
# 包含 protobuf 的头文件目录
# include_directories(${Protobuf_INCLUDE_DIRS})

# Add library NewtonSim
add_subdirectory("${PROJECT_SOURCE_DIR}/ext/NewtonSim")

# Add libaray booksim
add_subdirectory("${PROJECT_SOURCE_DIR}/ext/booksim")



file(GLOB_RECURSE SRC_FILES
        "${CMAKE_SOURCE_DIR}/src/*.hpp"
)

# set source and headers
file(GLOB_RECURSE OPERATION_SRC_FILES
        "${CMAKE_SOURCE_DIR}/src/operations/*.h"
        "${CMAKE_SOURCE_DIR}/src/operations/*.cc"
)


add_executable(
        NMC_Simulator src/main.cpp
        src/common_function.hpp
        src/dram_geometry.hpp

        src/operations/Operation.h
        src/operations/Operation.cc
        src/operations/PIMGEMV.h
        src/operations/PIMGEMV.cc
        src/operations/GEMV.h
        src/operations/GEMV.cc

        src/Client/Client.h
        src/Client/Client.cc
        src/Client/RequestGenerator.h
        src/Client/RequestGenerator.cc
        src/Client/BatchedRequest.h
        src/Client/BatchedRequest.cc

        src/tensor/BTensor.h
        src/tensor/BTensor.cc
        src/tensor/MyTensor.hpp
        src/tensor/MyTensor.cpp

        src/Scheduler/MyScheduler.hpp
        src/Scheduler/MyScheduler.cpp

        src/Interconnect/Interconnect.h
        src/Interconnect/Interconnect.cc
        src/Interconnect/MyInterconnect.hpp
        src/Interconnect/MyInterconnect.cpp

        src/Model/Model.cc

        src/Core/Core.h
        src/Core/Core.cc
        src/Core/Sram.h
        src/Core/Sram.cc
        src/Core/SystolicWS.cc
        src/Core/SystolicWS.h

        src/Core/MyCore.hpp
        src/Core/MyCore.cpp

        src/Program/StageProgram.h
        src/Program/StageProgram.cc

        src/DRAM/Dram.h
        src/DRAM/IDramBackend.h
        src/DRAM/Dram.cc
        src/DRAM/DramDataContainer.h
        src/DRAM/DramDataContainer.cpp

        src/DRAM/EventDrivenDram.h
        src/DRAM/EventDrivenDram.cpp

        ${OPERATION_SRC_FILES}

        src/simulator.cpp
)

target_link_libraries(NMC_Simulator dramsim3 booksim2)

include(CTest)

if(BUILD_TESTING)
    add_test(
        NAME dramsim3_build_output_location_test
        COMMAND ${CMAKE_COMMAND}
            -DDRAMSIM3_LIBRARY=$<TARGET_FILE:dramsim3>
            -DNEWTONSIM_SOURCE_DIR=${PROJECT_SOURCE_DIR}/ext/NewtonSim
            -P ${PROJECT_SOURCE_DIR}/tests/dramsim3_output_location_test.cmake
    )

    add_executable(
        data_container_test
        tests/data_container_test.cpp
        src/DRAM/DramDataContainer.cpp
    )
    target_include_directories(
        data_container_test
        PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/ext"
    )
    target_link_libraries(data_container_test PRIVATE dramsim3 format)
    add_test(NAME data_container_test COMMAND data_container_test)

    add_executable(
        data_container_dram_integration_test
        tests/data_container_dram_integration_test.cpp
        src/DRAM/Dram.cc
        src/DRAM/DramDataContainer.cpp
    )
    target_include_directories(
        data_container_dram_integration_test
        PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/ext"
        "${PROJECT_SOURCE_DIR}/ext/NewtonSim/include"
    )
    target_compile_definitions(
        data_container_dram_integration_test
        PRIVATE
        PH_SIM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(
        data_container_dram_integration_test
        PRIVATE
        dramsim3
        format
    )
    add_test(
        NAME data_container_dram_integration_test
        COMMAND data_container_dram_integration_test
    )

    add_executable(
        data_container_event_driven_integration_test
        tests/data_container_event_driven_integration_test.cpp
        src/DRAM/EventDrivenDram.cpp
        src/DRAM/DramDataContainer.cpp
    )
    target_include_directories(
        data_container_event_driven_integration_test
        PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/ext"
        "${PROJECT_SOURCE_DIR}/ext/NewtonSim/include"
    )
    target_compile_definitions(
        data_container_event_driven_integration_test
        PRIVATE
        PH_SIM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(
        data_container_event_driven_integration_test
        PRIVATE
        dramsim3
        format
    )
    add_test(
        NAME data_container_event_driven_integration_test
        COMMAND data_container_event_driven_integration_test
    )

    add_executable(
        virtual_memory_address_layout_test
        tests/virtual_memory_diagnostic.cpp
    )
    target_include_directories(
        virtual_memory_address_layout_test
        PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/ext"
    )
    target_compile_definitions(
        virtual_memory_address_layout_test
        PRIVATE
        PH_SIM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(
        virtual_memory_address_layout_test
        PRIVATE
        dramsim3
        format
    )
    add_test(
        NAME virtual_memory_address_layout_test
        COMMAND virtual_memory_address_layout_test
    )

    add_executable(
        pim_hash_address_group_test
        tests/pim_hash_address_group_test.cpp
    )
    target_include_directories(
        pim_hash_address_group_test
        PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/ext"
    )
    target_link_libraries(
        pim_hash_address_group_test
        PRIVATE
        dramsim3
        format
    )
    add_test(
        NAME pim_hash_address_group_test
        COMMAND pim_hash_address_group_test
    )

    add_test(
        NAME duplicate_dram_config_consistency_test
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/tests/duplicate_dram_config_consistency_test.cmake
    )

    add_executable(
        self_refresh_timing_config_test
        tests/self_refresh_timing_config_test.cpp
    )
    target_include_directories(
        self_refresh_timing_config_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/ext"
        "${PROJECT_SOURCE_DIR}/ext/NewtonSim/src"
    )
    target_compile_definitions(
        self_refresh_timing_config_test PRIVATE
        PH_SIM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(
        self_refresh_timing_config_test PRIVATE dramsim3 format
    )
    add_test(
        NAME self_refresh_timing_config_test
        COMMAND self_refresh_timing_config_test
    )

    add_executable(
        dram_geometry_config_test
        tests/dram_geometry_config_test.cpp
    )
    target_include_directories(
        dram_geometry_config_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/ext"
        "${PROJECT_SOURCE_DIR}/ext/NewtonSim/src"
    )
    target_compile_definitions(
        dram_geometry_config_test PRIVATE
        PH_SIM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(
        dram_geometry_config_test PRIVATE dramsim3 format
    )
    add_test(
        NAME dram_geometry_config_test
        COMMAND dram_geometry_config_test
    )

    file(MAKE_DIRECTORY
        "${CMAKE_BINARY_DIR}/test-output/data-container-vm-ca"
        "${CMAKE_BINARY_DIR}/test-output/data-container-vm-ed"
        "${CMAKE_BINARY_DIR}/test-output/data-container-gemm-ca"
        "${CMAKE_BINARY_DIR}/test-output/data-container-gemm-ed"
        "${CMAKE_BINARY_DIR}/test-output/base-gemm-ca"
        "${CMAKE_BINARY_DIR}/test-output/small-gemm-ca"
    )
    add_test(
        NAME data_container_virtual_memory_ca_smoke
        COMMAND NMC_Simulator
            -sc ../tests/configs/data_container_vm_ca_simulation.json
            -o "${CMAKE_BINARY_DIR}/test-output/data-container-vm-ca"
    )
    add_test(
        NAME data_container_virtual_memory_ed_smoke
        COMMAND NMC_Simulator
            -sc ../tests/configs/data_container_vm_ed_simulation.json
            -o "${CMAKE_BINARY_DIR}/test-output/data-container-vm-ed"
    )
    add_test(
        NAME data_container_gemm_ca_smoke
        COMMAND NMC_Simulator
            -sc ../tests/configs/data_container_gemm_ca_simulation.json
            -o "${CMAKE_BINARY_DIR}/test-output/data-container-gemm-ca"
    )
    add_test(
        NAME data_container_gemm_ed_smoke
        COMMAND NMC_Simulator
            -sc ../tests/configs/data_container_gemm_ed_simulation.json
            -o "${CMAKE_BINARY_DIR}/test-output/data-container-gemm-ed"
    )
    add_test(
        NAME base_gemm_cycle_accurate_smoke
        COMMAND NMC_Simulator
            -sc ../tests/fixtures/smoke/simulation_gemm_base_cycle_accurate.json
            -o "${CMAKE_BINARY_DIR}/test-output/base-gemm-ca"
    )
    add_test(
        NAME small_gemm_cycle_accurate_smoke
        COMMAND NMC_Simulator
            -sc ../tests/fixtures/smoke/simulation_gemm_small_cycle_accurate.json
            -o "${CMAKE_BINARY_DIR}/test-output/small-gemm-ca"
    )

    add_executable(
        nonvoid_control_flow_test
        tests/nonvoid_control_flow_test.cpp
    )
    target_include_directories(
        nonvoid_control_flow_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/ext"
    )
    target_link_libraries(nonvoid_control_flow_test PRIVATE dramsim3 format)
    add_test(NAME nonvoid_control_flow_test COMMAND nonvoid_control_flow_test)

    add_executable(
        statistics_edge_case_test
        tests/statistics_edge_case_test.cpp
    )
    target_include_directories(
        statistics_edge_case_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/ext"
    )
    target_link_libraries(statistics_edge_case_test PRIVATE dramsim3 format)
    add_test(NAME statistics_edge_case_test COMMAND statistics_edge_case_test)

    add_executable(
        runtime_guard_test
        tests/runtime_guard_test.cpp
    )
    target_include_directories(
        runtime_guard_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/ext"
    )
    target_link_libraries(runtime_guard_test PRIVATE dramsim3 format)
    add_test(NAME runtime_guard_test COMMAND runtime_guard_test)

    add_test(
        NAME dram_request_size_mismatch_test
        COMMAND ${CMAKE_COMMAND}
            -DSIMULATOR=$<TARGET_FILE:NMC_Simulator>
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${PROJECT_BINARY_DIR}
            -P ${PROJECT_SOURCE_DIR}/tests/dram_request_size_mismatch_test.cmake
    )
    set_tests_properties(dram_request_size_mismatch_test PROPERTIES TIMEOUT 30)

    add_test(
        NAME dram_channel_mismatch_test
        COMMAND ${CMAKE_COMMAND}
            -DSIMULATOR=$<TARGET_FILE:NMC_Simulator>
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${PROJECT_BINARY_DIR}
            -P ${PROJECT_SOURCE_DIR}/tests/dram_channel_mismatch_test.cmake
    )
    set_tests_properties(dram_channel_mismatch_test PROPERTIES TIMEOUT 30)

    add_test(
        NAME dram_frequency_mismatch_test
        COMMAND ${CMAKE_COMMAND}
            -DSIMULATOR=$<TARGET_FILE:NMC_Simulator>
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${PROJECT_BINARY_DIR}
            -P ${PROJECT_SOURCE_DIR}/tests/dram_frequency_mismatch_test.cmake
    )
    set_tests_properties(dram_frequency_mismatch_test PROPERTIES TIMEOUT 30)

    add_test(
        NAME full_simulation_smoke_test
        COMMAND ${CMAKE_COMMAND}
            -DSIMULATOR=$<TARGET_FILE:NMC_Simulator>
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${PROJECT_BINARY_DIR}
            -DTEST_MODE=full-simulation-smoke
            -P ${PROJECT_SOURCE_DIR}/tests/full_simulation_test.cmake
    )
    set_tests_properties(full_simulation_smoke_test PROPERTIES TIMEOUT 30)

    add_test(
        NAME ca_ed_consistency_test
        COMMAND ${CMAKE_COMMAND}
            -DSIMULATOR=$<TARGET_FILE:NMC_Simulator>
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${PROJECT_BINARY_DIR}
            -DTEST_MODE=ca-ed-consistency
            -P ${PROJECT_SOURCE_DIR}/tests/full_simulation_test.cmake
    )
    set_tests_properties(ca_ed_consistency_test PROPERTIES TIMEOUT 30)

    add_test(
        NAME iterative_decode_test
        COMMAND ${CMAKE_COMMAND}
            -DSIMULATOR=$<TARGET_FILE:NMC_Simulator>
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${PROJECT_BINARY_DIR}
            -DTEST_MODE=iterative-decode
            -P ${PROJECT_SOURCE_DIR}/tests/full_simulation_test.cmake
    )
    set_tests_properties(iterative_decode_test PROPERTIES TIMEOUT 150)

    add_executable(
        memory_access_owner_test
        tests/memory_access_owner_test.cpp
    )
    target_include_directories(
        memory_access_owner_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/ext"
    )
    target_link_libraries(memory_access_owner_test PRIVATE dramsim3 format)
    add_test(NAME memory_access_owner_test COMMAND memory_access_owner_test)

    add_test(
        NAME memory_access_lifecycle_gemm_test
        COMMAND ${CMAKE_COMMAND}
            -DSIMULATOR=$<TARGET_FILE:NMC_Simulator>
            -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${PROJECT_BINARY_DIR}
            -DTEST_MODE=memory-access-gemm
            -P ${PROJECT_SOURCE_DIR}/tests/full_simulation_test.cmake
    )
    set_tests_properties(memory_access_lifecycle_gemm_test PROPERTIES TIMEOUT 30)
endif()
