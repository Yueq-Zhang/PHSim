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
        src/DRAM/Dram.cc

        src/DRAM/EventDrivenDram.h
        src/DRAM/EventDrivenDram.cpp

        ${OPERATION_SRC_FILES}

        src/simulator.cpp
)

target_link_libraries(NMC_Simulator dramsim3 booksim2)

include(CTest)

if(BUILD_TESTING)
    add_executable(
        data_container_test
        tests/data_container_test.cpp
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
endif()
