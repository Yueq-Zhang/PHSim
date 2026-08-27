if(NOT DEFINED DRAMSIM3_LIBRARY OR DRAMSIM3_LIBRARY STREQUAL "")
    message(FATAL_ERROR "DRAMSIM3_LIBRARY is required")
endif()

if(NOT DEFINED NEWTONSIM_SOURCE_DIR OR NEWTONSIM_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "NEWTONSIM_SOURCE_DIR is required")
endif()

cmake_path(NORMAL_PATH DRAMSIM3_LIBRARY)
cmake_path(NORMAL_PATH NEWTONSIM_SOURCE_DIR)

if(NOT EXISTS "${DRAMSIM3_LIBRARY}")
    message(FATAL_ERROR
        "The dramsim3 target output does not exist: ${DRAMSIM3_LIBRARY}")
endif()

cmake_path(IS_PREFIX NEWTONSIM_SOURCE_DIR "${DRAMSIM3_LIBRARY}"
           NORMALIZE library_is_in_source_tree)
if(library_is_in_source_tree)
    message(FATAL_ERROR
        "dramsim3 must be emitted into the active build tree, not the "
        "NewtonSim source tree: ${DRAMSIM3_LIBRARY}")
endif()

message(STATUS "dramsim3 build output: ${DRAMSIM3_LIBRARY}")
