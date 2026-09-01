# PHSim development-test inventory

This file is an internal maintenance record. The root `README.md` intentionally
does not expose individual test cases as end-user workflows.

Current status: **development-only, keep until project feature freeze**.

Tests are compiled only when CMake is configured with
`-DBUILD_TESTING=ON`. Normal users can build the simulator with
`-DBUILD_TESTING=OFF` without compiling these targets.

## Registered CTest groups

The current CMake configuration registers 39 tests.

| Area | CTest entries | Main driver/assets | Freeze disposition |
| --- | --- | --- | --- |
| Build layout | `dramsim3_build_output_location_test` | `dramsim3_output_location_test.cmake` | Review after packaging is finalized. |
| DataContainer and address mapping | `data_container_test`, `data_container_dram_integration_test`, `data_container_event_driven_integration_test`, `virtual_memory_address_layout_test`, `pim_hash_address_group_test`, `data_container_virtual_memory_ca_smoke`, `data_container_virtual_memory_ed_smoke`, `data_container_gemm_ca_smoke`, `data_container_gemm_ed_smoke`, `memory_access_lifecycle_gemm_test` | DataContainer/DRAM C++ tests, `virtual_memory_diagnostic.cpp`, `pim_hash_address_group_test.cpp`, `tests/configs/`, and the full-simulation driver | Keep through DataContainer/VM stabilization; review redundant CA/ED fixture pairs first. |
| Configuration contracts | `duplicate_dram_config_consistency_test`, `self_refresh_timing_config_test`, `dram_geometry_config_test`, `dram_request_size_mismatch_test`, `dram_channel_mismatch_test`, `dram_frequency_mismatch_test`, `pim_bank_organization_mismatch_test` | Configuration C++ and CMake drivers plus mismatch fixtures | Recommended long-term guard set; prune only after equivalent startup validation is covered elsewhere. |
| Request generation and scheduling | `request_generator_test`, `trace_request_test`, `max_active_requests_test`, `iterative_decode_test`, `continuous_batching_test` | Request-generator C++ test and `full_simulation_test.cmake` with smoke fixtures | Keep through scheduler/continuous-batching stabilization. |
| Pruning composition | `pruning_tile_test`, `pruning_decode_fixed_test`, `pruning_decode_continuous_test`, `pruning_triple_test` | `pruning_composition_test.cmake`; 38 generated small CA/ED runs covering Tile/Decode Pruning with batching, VM, and DataContainer | Keep as a regression gate for active pruning, VM mapping-only replay, and the DataContainer exact-execution fallback. |
| Backend and model smoke coverage | `base_gemm_cycle_accurate_smoke`, `small_gemm_cycle_accurate_smoke`, `full_simulation_smoke_test`, `ca_ed_consistency_test` | GEMM/full-simulation fixtures and driver | Base/Small duplicate smoke cases are early pruning candidates once benchmark baselines move outside `tests/`; retain one end-to-end CA/ED gate. |
| Runtime safety | `nonvoid_control_flow_test`, `statistics_edge_case_test`, `runtime_guard_test`, `memory_access_owner_test` | Corresponding C++ tests | Small and inexpensive; recommended to retain unless static analysis or equivalent coverage replaces them. |
| Interconnect and BookSim | `simple_interconnect_backpressure_test`, `booksim2_unbounded_buffer_test`, `booksim2_finite_backpressure_test`, `booksim2_full_simulation_smoke_test` | `interconnect_backpressure_test.cpp`, BookSim configuration fixtures, and the full-simulation driver | Keep through BookSim/backpressure stabilization; later retain at least one finite-capacity and one end-to-end case. |

## Test-owned files

The following paths are development validation assets and should be reviewed
together rather than deleted individually:

- `tests/*.cpp`: direct unit/integration executables;
- `tests/*.cmake`: configuration and full-simulation drivers;
- `tests/configs/`: DataContainer, virtual-memory, and PIM-hash configurations;
- `tests/fixtures/smoke/`: small models, traces, DRAM/PIM configurations, and
  expected scheduling workloads;
- the `if(BUILD_TESTING)` section in the root `CMakeLists.txt`;
- `.github/workflows/ci.yml`, which invokes the Release CTest suite.

## Project-freeze cleanup gate

Do not delete tests merely because one development phase is complete. At final
project freeze, use this checklist:

1. Move any public example or benchmark that still references `tests/` into a
   stable `configs/` or `examples/` location.
2. Archive the final CA/ED, Mobile/Server, DataContainer, VM, continuous-batch,
   and BookSim reference results outside the build directory.
3. Keep a minimal release gate for configuration rejection, one full
   simulation, CA/ED logical-work parity, request ownership, continuous
   batching, DataContainer/VM, and finite-capacity interconnect behavior.
4. Remove only redundant diagnostic executables and duplicate fixtures whose
   behavior is already covered by the retained gate.
5. Update `CMakeLists.txt`, GitHub Actions, and this inventory in the same
   cleanup commit, then perform a clean Release build and simulation run.

Deleting the entire suite is possible after development, but is not
recommended for an open-source simulator: a small retained regression set is
the most direct way to detect compiler, dependency, and configuration drift.
