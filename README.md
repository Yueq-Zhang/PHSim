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
- Selectable fixed-latency or BookSim 2 mesh interconnect simulation, with
  optional finite input/output buffers and end-to-end backpressure (unbounded
  by default).
- NPU, NeuPIM, IANUS, and DASH memory-allocation schemes.
- PIM operations including GEMV, GWRITE, COMP, and READRES.
- Optional DRAM DataContainer support for maintaining simulated data values.
- Optional two-level virtual-to-physical address mapping and DRAM bank hashing.
- Output-token-driven Decode iteration and opt-in continuous batching with
  active-request backpressure, request-slot release, and stage-boundary
  rebatching. The historical fixed-stage scheduler remains the default.
- Tile Pruning with Loop-wise prediction or complete-work-unit Proportional
  sampling, including logical-work and timing compensation.
- Decode Pruning that samples repeated Decode operations and reuses their
  timing, traffic, workload, and DRAM-state templates in later iterations.

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build -j2
```

The simulator executable is generated at `build/NMC_Simulator`.

## Development validation

The test suite is intended for simulator development and continuous
integration; it is not part of the end-user workflow. Enable it only when
changing the implementation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The GitHub Actions workflow in `.github/workflows/ci.yml` repeats the Release
build and CTest suite on every push and pull request. It can also be started
manually from the Actions page.

## Run a simulation

Relative paths inside a simulation JSON are resolved against the process's
working directory. The normal cases under `configs/Cases/` are written for a
process launched from `build/`. Create the output directory, enter the build
directory, and launch the simulator from there:

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

Every normal run writes `_summary.tsv`, `core_timing.tsv`,
`request_stats.tsv`, `icnt_traffic.json`,
`interconnect_backpressure.json`, and `data_container_stats.json`. CycleAccurate runs also
write `dramsim3.json/.txt`; EventDriven runs write
`eventdrivendram.json/.txt`; BookSim runs additionally write
`booksim2_stats.json`.

For a field-by-field Chinese configuration reference, output/statistics
definitions, and copyable examples for full-model, single-operation, CA/ED,
DataContainer, and virtual-memory runs, see
[`docs/USER_GUIDE_zh.md`](docs/USER_GUIDE_zh.md).

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
| `allocation_scheme` | Select `NPU`, `NeuPIM`, `IANUS`, or `DASH` allocation |
| `max_batch_size` | Maximum requests included in one hardware Prefill/Decode program |
| `max_active_reqs` | Maximum requests resident in the scheduler/KV-cache set; additional Client requests are backpressured |
| `batch_scheduler` | Select `legacy` (default) or stage-boundary `continuous` batching |
| `virtual_mem_hash_enable` | Enable virtual-page mapping and DRAM bank hashing |
| `dram_data_container_enable` | Maintain simulated DRAM data values with instance-owned sparse storage |
| `dram_data_container_max_payload_mb` | Optional resident DataContainer payload limit in MiB; `0` means unlimited |
| `dram_trace_simulation_mode` | Use EventDriven DRAM when true; use the cycle-accurate backend when false |
| `accelerate_ctrl` | Enable simulator workload sampling |
| `accelerate_method` | Select the configured sampling method, such as `Loop_wise` or `Proportional` |
| `accelerate_sample_ratio` | Fraction of one complete K-loop sampled by Proportional GEMM/PIM-GEMV pruning; valid range `(0, 1]` |
| `attention_command_warmup_weight` | Weight assigned to attention warmup commands during Proportional command-count estimation; valid range `[0, 1]` |
| `softmax_warmup_rounds` | Complete per-core Softmax rounds retained before sampling; must be nonzero when Proportional Softmax pruning is used |
| `softmax_sample_rounds` | Complete per-core Softmax rounds sampled for prediction; must be nonzero when Proportional Softmax pruning is used |
| `compile_time_tile_pruning` | With Proportional sampling enabled, defer instruction construction for GEMM, attention-GEMM, and Softmax tiles that may be pruned |
| `decode_pruning_enabled` | Sample stable Decode operations and reuse their templates in later Decode iterations |
| `decode_pruning_iterations` | Repeated iterations constructed by the dedicated multi-layer `decode`/`npu_decode` path; must be greater than zero |
| `decode_pruning_sample_iterations` | Samples collected per eligible Decode operation before reuse; must be in `[1, decode_pruning_iterations]` |
| `test_single_op` | Run a configured single-operation workload |
| `test_multi_layer` | Run a configured multi-layer workload |
| `gen_request` | Generate requests instead of consuming the request trace |
| `gen_request_output_size` | Set the output-token target for fixed requests |
| `output_token_iteration_enable` | Repeat Decode to the output-token target; defaults to the legacy fixed stage sequence |

When `gen_request=false`, `request_file_path` points to a CSV with the schema:

```csv
input_tokens,output_tokens
4,3
4,1
4,2
```

For generated requests, `gen_request_count`, `gen_request_input_size`, and
`gen_request_output_size` provide the fixed request dimensions. The output
size controls the number of Decode rounds only when
`output_token_iteration_enable=true`.

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

### Important interconnect switches

| Key | Meaning |
| --- | --- |
| `icnt_type` | Select `simple` or `booksim2` |
| `icnt_latency` | Fixed-latency simple-interconnect delay |
| `icnt_ctrl_size` | BookSim control-only packet size in bytes; must be positive |
| `icnt_input_buffer_size` | Simple-interconnect input capacity in packets per node; `0` means unlimited |
| `icnt_output_buffer_size` | Simple-interconnect output capacity in packets per node; `0` means unlimited |
| `icnt_config_path` | Required BookSim `.icnt` configuration path when `icnt_type=booksim2` |

BookSim adapter capacities are configured in the `.icnt` file with
`input_buffer_size` (flits/source), `ejection_buffer_size`
(flits/destination/VC), and `boundary_buffer_size`
(complete packets/destination/VC). Zero or omission keeps each adapter queue
unbounded; `vc_buf_size` remains the positive per-router VC capacity.

### Typical configuration recipes

#### Tile Pruning

Tile Pruning reduces simulation work inside a large operation while preserving
its logical workload. `Loop_wise` predicts repeated loop timing after the
initial loop samples. `Proportional` keeps complete boundaries: K-loop units for
GEMM/PIM-GEMV, head rounds for attention GEMM, and outer rounds for Softmax.
Irregular tail tiles are still simulated instead of being extrapolated.

The following configuration enables Proportional Tile Pruning and avoids
building instructions eagerly for eligible GEMM, attention-GEMM, and Softmax
tiles:

```json
{
  "accelerate_ctrl": true,
  "accelerate_method": "Proportional",
  "accelerate_sample_ratio": 0.25,
  "attention_command_warmup_weight": 0.0,
  "softmax_warmup_rounds": 2,
  "softmax_sample_rounds": 4,
  "compile_time_tile_pruning": true
}
```

When the selected operation is too small to provide complete warmup, sample,
and prediction regions, PHSim disables Proportional pruning for that operation
and simulates it normally. Estimated tiles, calculations, memory traffic,
interconnect traffic, DRAM commands, and timing are reported separately from
measured work and are also included in the logical totals.

#### Decode Pruning

Decode Pruning operates across repeated Decode iterations. It first executes
and samples each eligible operation, then skips instruction materialization and
execution for later occurrences with the same operation name, request-batch
signature, tile count, and loop shape. The sampled core timing, logical
workload, interconnect traffic, DRAM commands, and DRAM state are applied to
both CycleAccurate and EventDriven runs.

Without virtual memory, the target set contains Decode FFN operations and the
attention Q, K, V, and projection generation operations. QK, SV, and Softmax
remain explicitly simulated because their work changes with KV-cache length.
With virtual memory enabled, only the address-stable FFN subset is pruned;
attention generation remains explicitly simulated because KV-cache growth can
change its logical addresses between token iterations.

For normal full-model output-token iteration, enable:

```json
{
  "test_single_op": false,
  "test_multi_layer": false,
  "output_token_iteration_enable": true,
  "decode_pruning_enabled": true,
  "decode_pruning_iterations": 8,
  "decode_pruning_sample_iterations": 1
}
```

In a normal full-model run, the request's output-token target determines the
number of Decode iterations; `decode_pruning_iterations` controls only the
dedicated multi-layer `decode`/`npu_decode` path. It must nevertheless be at
least `decode_pruning_sample_iterations` because configuration validation
enforces that relationship.

For a full-fidelity baseline, set both `accelerate_ctrl=false` and
`decode_pruning_enabled=false`. Compare this baseline against pruned runs using
logical totals as well as the measured/estimated breakdown.

#### Continuous batching

To enable output-token iteration with stage-boundary continuous batching, set
the following fields in a complete inference configuration:

```json
{
  "max_batch_size": 2,
  "max_active_reqs": 2,
  "batch_scheduler": "continuous",
  "output_token_iteration_enable": true,
  "test_single_op": false,
  "test_multi_layer": false
}
```

Completed requests release their KV-cache and active-request slot immediately.
New requests may enter at the next stage boundary. Prefill and Decode are not
mixed in one hardware program. Batch capacity is determined only by the number
of requests through `min(max_batch_size, max_active_reqs)`.

#### DataContainer and virtual memory

To maintain sparse DRAM payloads after virtual-page translation and address
hashing, enable:

```json
{
  "virtual_mem_hash_enable": true,
  "dram_data_container_enable": true,
  "dram_data_container_max_payload_mb": 64
}
```

DataContainer tracks transparent bytes and data movement; it does not
numerically evaluate GEMM/GEMV operations. Start with a small model and a
finite payload limit before scaling the workload.

DataContainer payload mutation is not reconstructed from pruning statistics.
When DataContainer is enabled, PHSim therefore disables Tile/other simulation
acceleration and Decode Pruning and emits a warning; this is an intentional
exact-execution fallback.

Virtual memory supports Proportional Tile Pruning and Decode Pruning through a
mapping-only replay path. Skipped tiles are materialized only to reproduce each
logical-to-physical mapping side effect; they are not injected into the core,
interconnect, or DRAM backend. Decode replay is restricted to FFN operations,
while address-varying attention generation executes normally. Other Tile
Pruning methods (`naive` and `Loop_wise`) still fall back to exact execution
when virtual memory is enabled. The regression suite compares mapping counts,
per-channel distribution, address-pair fingerprints, and the final page table
against an unpruned baseline.

#### BookSim interconnect

To select BookSim, set the compute configuration to:

```json
{
  "icnt_type": "booksim2",
  "icnt_config_path": "../configs/booksim2_configs/mesh_2x2.icnt",
  "icnt_ctrl_size": 8
}
```

Adjust the path to match the process working directory and the selected
topology. A normal BookSim run should finish with equal injected/ejected packet
and payload-byte counts in `booksim2_stats.json`.

## Repository layout

```text
configs/        System, DRAM, PIM, model, and inference configurations
ext/            Bundled third-party dependencies
sample_trace/   Small DRAM trace example
src/            Simulator implementation
tests/          Developer-only validation assets
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
  so set `dram_data_container_max_payload_mb` and start with a small case
  before running large models.
- `batch_scheduler` defaults to `legacy`, so adding the new simulator version
  without adding the field does not change an existing case's stage sequence.
- Continuous batching requires `test_single_op=false`,
  `test_multi_layer=false`, and `output_token_iteration_enable=true`.
- Compare CA and ED first by completed stages, outstanding-request count, and
  logical interconnect traffic. Their detailed command timing is intentionally
  backend-specific.

## License

Original PHSim code is distributed under the [MIT License](LICENSE).
Bundled dependencies retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
