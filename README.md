# PHSim

PHSim is a research simulator for transformer inference on accelerator systems
with conventional DRAM and processing-in-memory (PIM) support. It models the
compute cores, interconnect, memory allocation, transformer operations, and two
DRAM execution backends.

PHSim is an independent open-source research project. It is not an official
Intel product and is not affiliated with or endorsed by Intel.

## Features

- Cycle-accurate DRAM simulation through the bundled NewtonSim backend.
- Event-driven DRAM simulation for faster design-space exploration.
- NPU, IANUS, and DASH memory-allocation schemes.
- PIM operations including GEMV, GWRITE, COMP, and READRES.
- Optional DRAM DataContainer support for maintaining simulated data values.
- Optional two-level virtual-to-physical address mapping and DRAM bank hashing.
- Loop-wise and proportional workload sampling for accelerated simulation.
- Small CTest cases covering DataContainer, virtual-memory layout, and PIM
  hash-address grouping.

## Supported environment

The currently validated environment is Linux or WSL2 with:

- CMake 3.22 or newer
- GCC 11.4 or newer
- A C++17-capable standard library
- BLAS and Boost development headers (for Ubuntu: `libblas-dev libboost-dev`)

Native Windows/MSVC builds are not currently part of the validated workflow.

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j2
```

The simulator executable is generated at `build/NMC_Simulator`.

## Run the tests

```bash
ctest --test-dir build --output-on-failure
```

The GitHub Actions workflow in `.github/workflows/ci.yml` repeats the Release
build and CTest suite on every push and pull request. It can also be started
manually from the Actions page.

The current test suite contains:

- `data_container_test`
- `data_container_dram_integration_test`
- `data_container_event_driven_integration_test`
- `virtual_memory_address_layout_test`
- `pim_hash_address_group_test`
- `data_container_virtual_memory_ca_smoke`
- `data_container_virtual_memory_ed_smoke`
- `data_container_gemm_ca_smoke`
- `data_container_gemm_ed_smoke`
- `nonvoid_control_flow_test`
- `statistics_edge_case_test`
- `runtime_guard_test`
- `full_simulation_smoke_test`
- `ca_ed_consistency_test`
- `memory_access_owner_test`
- `memory_access_lifecycle_gemm_test`

The two full-simulation tests use a dedicated 64-dimensional, four-token Add
workload. They exercise the main executable through the scheduler, core,
interconnect, and both DRAM backends without the memory and runtime cost of a
full transformer workload. The CA/ED test requires identical logical traffic
and completion counts while allowing backend timing and write-command issue
counts to differ.

The MemoryAccess tests verify unique request ownership, reusable owner slots,
zero outstanding requests at normal completion, and a fixed single-operation
GEMM baseline on both CycleAccurate and EventDriven DRAM.

## Run a simulation

Configuration paths in the bundled case files are relative to the build
directory. Create the output directory, enter the build directory, and launch
the simulator from there:

```bash
mkdir -p output/nano
cd build
./NMC_Simulator \
  --simulation_config ../configs/Cases/Nano/simulation_config_Nano.json \
  --output_path ../output/nano
```

Equivalent short options are `-sc` and `-o`.

The simulator removes an existing `log.txt` in the selected output directory,
initializes the configured accelerator and memory system, runs the model, and
writes statistics to that directory.

## Configuration structure

A top-level `simulation_config_*.json` file connects the other configuration
layers:

| Key | Purpose |
| --- | --- |
| `compute_die_config_file_path` | Compute-core and interconnect configuration |
| `DRAM_config_file_path` | DRAM organization, timing, and energy parameters |
| `PIM_config_file_path` | PIM architecture and command parameters |
| `model_config_file_path` | Transformer model dimensions |
| `inference_config_file_path` | Workload, mapping, and acceleration controls |
| `request_file_path` | Request trace used when generated requests are disabled |

Ready-to-edit system cases are provided under `configs/Cases/` for Nano, Tiny,
Small, Base, Mobile, and Server configurations.

### Important inference switches

| Key | Meaning |
| --- | --- |
| `allocation_scheme` | Select `NPU`, `IANUS`, or `DASH` allocation |
| `virtual_mem_hash_enable` | Enable virtual-page mapping and DRAM bank hashing |
| `dram_data_container_enable` | Maintain simulated DRAM data values with instance-owned sparse storage |
| `dram_data_container_max_payload_mb` | Optional resident DataContainer payload limit in MiB; `0` means unlimited |
| `dram_trace_simulation_mode` | Use EventDriven DRAM when true; use the cycle-accurate backend when false |
| `accelerate_ctrl` | Enable simulator workload sampling |
| `accelerate_method` | Select the configured sampling method, such as `Loop_wise` or `Proportional` |
| `test_single_op` | Run a configured single-operation workload |
| `test_multi_layer` | Run a configured multi-layer workload |
| `gen_request` | Generate requests instead of consuming the request trace |

Address values passed through the DRAM-facing simulator interfaces use one
address unit per complete DRAM burst. When virtual memory is enabled, the
mapping pipeline is:

```text
virtual burst address
  -> logical page and page offset
  -> physical page allocation
  -> in-page bank hash
  -> channel/rank/bank-group/bank/row/column fields
```

## Repository layout

```text
configs/        System, DRAM, PIM, model, and inference configurations
ext/            Bundled third-party dependencies
sample_trace/   Small DRAM trace example
src/            Simulator implementation
tests/          Small regression and integration tests
```

## Reproducibility notes

- Keep the DRAM organization fixed when comparing allocation schemes or DRAM
  backends.
- Compare CycleAccurate and EventDriven results with command-level counters in
  addition to final completion counts.
- Distinguish measured, estimated, and logical work when simulation sampling is
  enabled.
- DataContainer materializes only touched DRAM columns; missing locations read
  as zero. Large real-data workloads can still consume significant host memory,
  so set `dram_data_container_max_payload_mb` and start with the bundled small
  tests before running large models.

## License

Original PHSim code is distributed under the [MIT License](LICENSE).
Bundled dependencies retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
