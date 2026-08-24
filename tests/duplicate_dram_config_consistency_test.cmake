if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(PRIMARY_CONFIG_DIR "${SOURCE_DIR}/configs/memory_config")
set(NEWTONSIM_CONFIG_DIR "${SOURCE_DIR}/ext/NewtonSim/configs")

function(parse_ini CONFIG_PATH CONFIG_ROLE CONFIG_NAME OUTPUT_VAR)
    file(STRINGS "${CONFIG_PATH}" RAW_LINES)
    set(CURRENT_SECTION "")
    set(SEEN_KEYS "")
    set(NORMALIZED_ENTRIES "")
    set(SAW_HBM_DUAL_CMD FALSE)

    foreach(RAW_LINE IN LISTS RAW_LINES)
        string(REGEX REPLACE ";.*$" "" LINE "${RAW_LINE}")
        string(STRIP "${LINE}" LINE)
        if(LINE STREQUAL "")
            continue()
        endif()

        if(LINE MATCHES "^\\[([^]]+)\\]$")
            set(CURRENT_SECTION "${CMAKE_MATCH_1}")
            continue()
        endif()

        if(NOT LINE MATCHES "^([^=]+)=(.*)$")
            message(FATAL_ERROR
                "Unrecognized INI line in ${CONFIG_PATH}: ${LINE}")
        endif()

        set(KEY "${CMAKE_MATCH_1}")
        set(VALUE "${CMAKE_MATCH_2}")
        string(STRIP "${KEY}" KEY)
        string(STRIP "${VALUE}" VALUE)
        set(QUALIFIED_KEY "${CURRENT_SECTION}.${KEY}")

        list(FIND SEEN_KEYS "${QUALIFIED_KEY}" DUPLICATE_INDEX)
        if(NOT DUPLICATE_INDEX EQUAL -1)
            message(FATAL_ERROR
                "Duplicate INI key ${QUALIFIED_KEY} in ${CONFIG_PATH}")
        endif()
        list(APPEND SEEN_KEYS "${QUALIFIED_KEY}")

        if(KEY STREQUAL "tRTP_L" OR KEY STREQUAL "tRTP_S" OR
                KEY STREQUAL "tCKSRE")
            message(FATAL_ERROR
                "Deprecated ignored key ${KEY} found in ${CONFIG_PATH}; use the canonical timing key")
        endif()

        # These two same-name files predate this consistency test with narrow,
        # known differences. Validate the exact difference and compare every
        # other effective field normally.
        if(CONFIG_NAME STREQUAL "HBM_4Gb_x128.ini"
                AND QUALIFIED_KEY STREQUAL "dram_structure.hbm_dual_cmd")
            set(SAW_HBM_DUAL_CMD TRUE)
            string(TOLOWER "${VALUE}" LOWER_VALUE)
            if(NOT CONFIG_ROLE STREQUAL "NEWTONSIM" OR
                    NOT LOWER_VALUE STREQUAL "false")
                message(FATAL_ERROR
                    "Unexpected hbm_dual_cmd variant in ${CONFIG_PATH}")
            endif()
            continue()
        endif()

        if(CONFIG_NAME STREQUAL "HBM2_8Gb_x128_dram.ini"
                AND QUALIFIED_KEY STREQUAL "system.channels")
            if(CONFIG_ROLE STREQUAL "PRIMARY")
                set(EXPECTED_CHANNELS 4)
            else()
                set(EXPECTED_CHANNELS 32)
            endif()
            if(NOT VALUE STREQUAL "${EXPECTED_CHANNELS}")
                message(FATAL_ERROR
                    "Expected channels=${EXPECTED_CHANNELS} in ${CONFIG_PATH}, got ${VALUE}")
            endif()
            set(VALUE "<validated-variant>")
        endif()

        list(APPEND NORMALIZED_ENTRIES "${QUALIFIED_KEY}=${VALUE}")
    endforeach()

    if(CONFIG_NAME STREQUAL "HBM_4Gb_x128.ini")
        if(CONFIG_ROLE STREQUAL "NEWTONSIM" AND NOT SAW_HBM_DUAL_CMD)
            message(FATAL_ERROR
                "Expected hbm_dual_cmd=False in ${CONFIG_PATH}")
        elseif(CONFIG_ROLE STREQUAL "PRIMARY" AND SAW_HBM_DUAL_CMD)
            message(FATAL_ERROR
                "Primary ${CONFIG_PATH} must not define hbm_dual_cmd")
        endif()
    endif()

    list(SORT NORMALIZED_ENTRIES)
    string(JOIN "\n" NORMALIZED_TEXT ${NORMALIZED_ENTRIES})
    set(${OUTPUT_VAR} "${NORMALIZED_TEXT}" PARENT_SCOPE)
endfunction()

# Validate every source INI, including Case-only configurations that have no
# NewtonSim mirror.
file(GLOB_RECURSE ALL_SOURCE_INIS
    "${SOURCE_DIR}/configs/Cases/*.ini"
    "${SOURCE_DIR}/configs/memory_config/*.ini"
    "${SOURCE_DIR}/ext/NewtonSim/configs/*.ini"
)
foreach(CONFIG_PATH IN LISTS ALL_SOURCE_INIS)
    get_filename_component(CONFIG_NAME "${CONFIG_PATH}" NAME)
    string(FIND "${CONFIG_PATH}" "${NEWTONSIM_CONFIG_DIR}/" NEWTONSIM_PREFIX)
    string(FIND "${CONFIG_PATH}" "${PRIMARY_CONFIG_DIR}/" PRIMARY_PREFIX)
    if(NEWTONSIM_PREFIX EQUAL 0)
        set(CONFIG_ROLE "NEWTONSIM")
    elseif(PRIMARY_PREFIX EQUAL 0)
        set(CONFIG_ROLE "PRIMARY")
    else()
        set(CONFIG_ROLE "CASE")
    endif()
    parse_ini("${CONFIG_PATH}" "${CONFIG_ROLE}" "${CONFIG_NAME}" UNUSED_NORMALIZED)
endforeach()

# Compare all same-name primary/NewtonSim pairs by effective section/key/value,
# ignoring comments, whitespace, ordering, and the validated exceptions above.
file(GLOB PRIMARY_CONFIGS "${PRIMARY_CONFIG_DIR}/*.ini")
set(COMMON_CONFIG_COUNT 0)
foreach(PRIMARY_CONFIG IN LISTS PRIMARY_CONFIGS)
    get_filename_component(CONFIG_NAME "${PRIMARY_CONFIG}" NAME)
    set(NEWTONSIM_CONFIG "${NEWTONSIM_CONFIG_DIR}/${CONFIG_NAME}")
    if(NOT EXISTS "${NEWTONSIM_CONFIG}")
        continue()
    endif()

    math(EXPR COMMON_CONFIG_COUNT "${COMMON_CONFIG_COUNT} + 1")
    parse_ini("${PRIMARY_CONFIG}" "PRIMARY" "${CONFIG_NAME}" PRIMARY_NORMALIZED)
    parse_ini("${NEWTONSIM_CONFIG}" "NEWTONSIM" "${CONFIG_NAME}" NEWTONSIM_NORMALIZED)
    if(NOT PRIMARY_NORMALIZED STREQUAL NEWTONSIM_NORMALIZED)
        message(FATAL_ERROR
            "Duplicate DRAM configs differ: ${PRIMARY_CONFIG} and ${NEWTONSIM_CONFIG}")
    endif()
endforeach()

if(COMMON_CONFIG_COUNT EQUAL 0)
    message(FATAL_ERROR "No duplicate DRAM configuration pairs were checked")
endif()

message(STATUS
    "Validated ${COMMON_CONFIG_COUNT} duplicate DRAM configuration pairs")
