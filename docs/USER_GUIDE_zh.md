# PHSim 配置、输出与典型用例指南

本文档对应当前仓库中的 `NMC_Simulator` 可执行程序，面向需要修改系统配置、运行实验和解释结果的使用者。配置与命令以 Linux/WSL2 Release 构建为准。

> 重要边界：PHSim 是时序、流量、能耗与数据移动仿真器，不执行训练框架意义上的真实 Transformer 数值计算。DataContainer 可以维护仿真中的字节数据，但不会把 PHSim 变成 PyTorch/TensorFlow 推理引擎。

## 1. 快速开始

在仓库根目录构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

配置文件中的相对路径按照程序启动时的工作目录解析。仓库提供的 Case 都以 `build/` 为工作目录，因此运行前应创建输出目录并进入 `build/`：

```bash
mkdir -p output/nano
cd build
./NMC_Simulator \
  --simulation_config ../configs/Cases/Nano/simulation_config_Nano.json \
  --output_path output/nano
```

命令行参数只有两个：

| 参数 | 短参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--simulation_config` | `-sc` | 路径字符串 | 空字符串 | 顶层仿真配置。实际运行必须提供。 |
| `--output_path` | `-o` | 路径字符串 | `output` | 输出目录。目录必须提前创建。已有 `log.txt` 会在启动时删除。 |

## 2. 配置层次与字段判定规则

顶层配置把五类硬件/工作负载配置连接起来：

```text
simulation_config_*.json
  ├─ compute-die JSON
  ├─ DRAM INI
  ├─ PIM JSON
  ├─ model JSON
  ├─ inference JSON
  └─ request trace（仅 gen_request=false 时读取）
```

本文使用以下标记：

- **必填**：代码通过 `json["field"]` 读取，没有可用的代码默认值；建议始终显式填写。
- **默认值 X**：代码通过 `json.value("field", X)` 或 `INIReader(..., X)` 读取。
- **条件必填**：只有选择特定后端/模式时才使用。
- **保留字段**：样例中存在，但当前初始化路径没有消费；修改它不会可靠地改变仿真。

JSON 数字必须非负并适合对应的 C++ 无符号类型。除代码明确检查的字段外，当前解析器对很多“必须大于零”的条件没有统一报错；下表中的约束应视为有效配置要求。

## 3. JSON 配置参数手册

### 3.1 顶层 simulation JSON

示例：`configs/Cases/Nano/simulation_config_Nano.json`。

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `compute_die_config_file_path` | 字符串/路径 | 必填 | 计算核心、向量核、片上存储与互连配置。 |
| `DRAM_config_file_path` | 字符串/路径 | 必填 | DRAM组织、时序、功耗和地址布局INI。 |
| `PIM_config_file_path` | 字符串/路径 | 必填 | PIM结构、缓冲、面积与功耗参数JSON。 |
| `model_config_file_path` | 字符串/路径 | 必填 | 模型维度JSON。 |
| `inference_config_file_path` | 字符串/路径 | 必填 | 请求、映射、后端和加速开关JSON。 |
| `request_file_path` | 字符串/路径 | 必填字段 | `gen_request=false` 时必须指向可读请求数据；`gen_request=true` 时不会打开该文件。 |

路径应与启动目录一致。使用仓库 Case 时，从 `build/` 启动并保留 `../configs/...` 形式。

### 3.2 计算芯片 JSON

示例：`configs/Cases/Nano/Nano_1x64x64.json`。

#### 核心与向量单元

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `num_cores` | 无符号整数/个 | 必填 | 核心数，必须大于0。 |
| `core_type` | 字符串 | 必填 | 仅支持 `systolic_ws`、`systolic_os`。 |
| `core_freq` | 无符号整数/MHz | 必填 | 核心频率，必须大于0；用于与DRAM、互连时钟换算。 |
| `core_width` | 无符号整数/PE列 | 必填 | 脉动阵列宽度；当前地址分配要求偶数。 |
| `core_height` | 无符号整数/PE行 | 必填 | 脉动阵列高度。 |
| `vector_core_count` | 无符号整数/个 | 必填 | 每个核心的向量单元数量。 |
| `vector_core_width` | 无符号整数/元素 | 必填 | 向量并行宽度。 |
| `add_latency` | 无符号整数/核心周期 | 必填 | 向量加法延迟。 |
| `mul_latency` | 无符号整数/核心周期 | 必填 | 向量乘法延迟。 |
| `exp_latency` | 无符号整数/核心周期 | 必填 | 指数运算延迟。 |
| `gelu_latency` | 无符号整数/核心周期 | 必填 | GELU运算延迟。 |
| `add_tree_latency` | 无符号整数/核心周期 | 必填 | 归约加法树延迟。 |
| `scalar_sqrt_latency` | 无符号整数/核心周期 | 必填 | 标量平方根延迟。 |
| `scalar_add_latency` | 无符号整数/核心周期 | 必填 | 标量加法延迟。 |
| `scalar_mul_latency` | 无符号整数/核心周期 | 必填 | 标量乘法延迟。 |

#### 片上存储、互连与数据格式

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `spad_size` | 无符号整数/KiB | 必填 | 主scratchpad容量；多数算子按一半容量预留双缓冲。 |
| `accum_spad_size` | 无符号整数/KiB | 必填 | 累加scratchpad容量。 |
| `sram_width` | 无符号整数/bit | 必填 | SRAM接口宽度。 |
| `icnt_type` | 字符串 | 必填 | `simple` 或 `booksim2`。 |
| `icnt_latency` | 无符号整数/互连周期 | 条件必填 | `simple`互连延迟。代码允许字段缺失，但没有可靠的显式默认值，建议始终填写。 |
| `icnt_freq` | 无符号整数/MHz | 必填 | 互连频率，必须大于0。 |
| `icnt_config_path` | 字符串/路径 | 条件必填 | `booksim2`配置路径；`simple`模式不使用。 |
| `precision` | 无符号整数/byte/element | 必填 | 旧的统一数据宽度，仍被部分算子和SRAM使用；例如FP16填`2`。 |
| `precision_weight` | 无符号整数/byte/element | 必填 | 权重元素字节数，必须大于0。 |
| `precision_activation` | 无符号整数/byte/element | 必填 | 激活元素字节数，必须大于0。 |
| `precision_cache` | 无符号整数/byte/element | 必填 | KV Cache元素字节数，必须大于0。 |
| `precision_psum` | 无符号整数/byte/element | 必填 | 部分和元素字节数，必须大于0。 |
| `layout` | 字符串 | 必填 | 当前样例使用 `NHWC`；主要作为布局标签传递。 |
| `scheduler` | 字符串 | 必填 | 当前样例使用 `simple`。 |
| `operation_log_output_path` | 字符串/路径 | 必填字段 | 兼容字段；主输出仍由命令行 `--output_path` 和内部 `log_dir` 决定。 |

以下字段出现在部分样例中，但当前初始化代码不读取：

| 字段 | 状态 | 建议 |
| --- | --- | --- |
| `__sram_size` | 注释/元数据字段 | 可保留，不参与仿真。 |
| `sram_size` | 保留字段 | 当前使用 `spad_size` 和 `accum_spad_size`。 |
| `process_bit` | 保留字段 | 当前有效精度由 `precision*` 的字节数控制。 |

### 3.3 推理与实验控制 JSON

示例：`configs/Cases/inference_config.json`。

#### 请求规模与地址映射

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `max_batch_size` | 无符号整数/请求 | 必填 | 调度批次上限，必须大于0。 |
| `max_active_reqs` | 无符号整数/请求 | 必填 | 同时驻留的最大请求数；参与KV Cache容量预留。 |
| `max_seq_len` | 无符号整数/token | 必填 | 最大序列长度；参与KV Cache容量预留。 |
| `kv_cache_entry_size` | 无符号整数/token | 必填 | IANUS/DASH中每个Cache entry覆盖的token数；NPU布局会根据DRAM页宽重新推导。 |
| `allocation_scheme` | 字符串 | 必填 | 支持 `NPU`、`NeuPIM`、`IANUS`、`DASH`，区分大小写。 |
| `virtual_mem_hash_enable` | 布尔 | `false` | 打开两级虚拟页映射和页内bank hash。 |
| `dram_data_container_enable` | 布尔 | `false` | 打开实例级稀疏DRAM数据存储；CA和ED均接入。 |
| `dram_data_container_max_payload_mb` | 无符号整数/MiB | `0` | 已物化payload上限；`0`表示不设上限。达到上限时写入抛出异常。建议大模型先设小值。 |

虚拟内存实现固定使用 **4 MiB页** 和 **1 KiB页内hash粒度**。DRAM接口地址单位是一整个DRAM burst，而不是byte。开启映射时要求：

- `request_size_bytes = bus_width / 8 × BL`；
- 4 MiB页和1 KiB hash单元都能被该burst字节数整除；
- 总物理容量能被burst和4 MiB页整除；
- channels、ranks、bankgroups、banks、rows、columns等位切分维度应为2的幂；
- PIM JSON的 `dram_channels` 必须和INI的 `channels` 一致。

映射过程为：

```text
虚拟burst地址
  -> 4 MiB逻辑页号 + 页内偏移
  -> 首次访问时确定性分配物理页
  -> 以1 KiB为粒度改写页内channel/rank/bank-group/bank字段
  -> 生成物理burst地址
```

#### 测试模式与加速模式

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `test_single_op` | 布尔 | 必填 | `true`时只构造单算子程序，并优先于多层模式。 |
| `test_single_op_name` | 字符串 | 必填 | 支持 `rmsnorm`、`layernorm`、`gemm`、`gemm_att`、`gemv`、`gemv_att`、`softmax`、`add`、`mul`、`gelu`、`silu`、`data_convert`/`dataconvert`、`pim_gemv`、`pim_gemv_qkt`、`pim_gemv_sv` 等注册名称。 |
| `test_multi_layer` | 布尔 | 必填 | `true`时构造预设的算子块/单层路径。 |
| `test_multi_layer_name` | 字符串 | 必填 | 当前实现包含 `ffn`、`decode_ffn`、`decode_attn`、`attn`、`decode`、`npu_decode`、`prefill`、`llama_ffn`、`data_convert_weights`、`llama_decode_ffn`、`llama_attn`、`llama_decode_attn`。 |
| `accelerate_ctrl` | 布尔 | `false` | 开启工作量采样/预测。做基线或CA/ED一致性验证时应关闭。 |
| `accelerate_method` | 字符串 | 空字符串 | 当前实现识别 `naive`、`Loop_wise`、`Proportional`。 |
| `accelerate_sample_ratio` | 浮点/比例 | `0.25` | Proportional采样比例；有效范围应为 `(0,1]`，具体算子还要求存在足够完整循环。 |
| `attention_command_warmup_weight` | 浮点/比例 | `0.0` | 注意力命令补偿的warmup权重；代码强制 `[0,1]`。 |
| `softmax_warmup_rounds` | 无符号整数/轮 | `2` | Proportional Softmax warmup轮数；相关加速时必须大于0。 |
| `softmax_sample_rounds` | 无符号整数/轮 | `4` | Proportional Softmax采样轮数；相关加速时必须大于0。 |
| `compile_time_tile_pruning` | 布尔 | `false` | 在程序构造期裁剪可预测tile。 |
| `decode_pruning_enabled` | 布尔 | `false` | 打开decode迭代采样/复用。 |
| `decode_pruning_iterations` | 无符号整数/迭代 | `1` | 必须大于0。 |
| `decode_pruning_sample_iterations` | 无符号整数/迭代 | `1` | 必须大于0且不能超过 `decode_pruning_iterations`。 |
| `dram_trace_simulation_mode` | 布尔 | `false` | `false` = CycleAccurate/NewtonSim；`true` = EventDriven。 |
| `record_dram_completion_trace` | 布尔 | `false` | 仅在编译时 `ENABLE_DRAM_ALIGNMENT_TRACE=1` 时生效；默认Release构建不会生成该调试trace。 |

#### 请求生成

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `gen_request` | 布尔 | 必填 | `true`使用合成请求；`false`读取顶层 `request_file_path`。 |
| `gen_request_count` | 无符号整数/请求 | 必填 | 合成请求数量。当前部分地址交织表针对1、2、4请求做了预设，其他值会回退。 |
| `gen_request_input_size` | 无符号整数/token | 必填 | 固定合成输入长度。 |
| `gen_request_output_size` | 无符号整数/token | `0` | 固定请求的目标输出token数。`gen_random_request=true`时由代码内随机输出范围覆盖；开启输出token迭代时，每个请求会以其实际 `output_size` 作为终止条件。 |
| `output_token_iteration_enable` | 布尔 | `false` | `false`保留原有固定stage序列；`true`时仅在完整模型模式下执行一次Prefill，再重复所选Decode后端，直到 `generated == output_size`。单算子/多层测试会忽略该选项并打印警告。 |
| `gen_random_request` | 布尔 | 必填 | `true`时使用代码内固定随机范围：输入 `[128,256)`、输出 `[256,512)`。 |
| `request_interval` | 无符号整数/核心周期 | 必填 | 请求到达间隔参数；必须大于0。 |

当 `test_single_op=false` 且 `test_multi_layer=false` 时，调度器进入完整模型阶段路径；最终执行多少层由 `model_n_layer` 决定。当前仓库的模型样例都把 `model_n_layer` 设为1，以控制仿真规模。

输出token迭代的Decode后端由 `allocation_scheme` 决定：`NPU`选择 `NPU_Decode`（SA），其他当前支持的布局选择 `Decode`（PIM）。开启时要求每个实际请求满足 `input_size + output_size <= max_seq_len`，否则在请求生成时终止并报告配置错误。请求完成前KV Cache会跨Decode轮次保留，达到目标后才释放。

### 3.4 模型 JSON

示例：`configs/model_config/opt-350M.json`。

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `model_name` | 字符串 | 必填 | 模型名称；名称包含 `llama`（大小写变体）时选择Llama类算子图。 |
| `model_params_b` | 浮点/十亿参数 | 必填 | 元数据/报告字段。 |
| `model_vocab_size` | 无符号整数/token | 必填 | 词表大小。 |
| `model_n_layer` | 无符号整数/层 | 必填 | 构造的Transformer层数。大于1会显著增加运行时间和内存。 |
| `model_n_head` | 无符号整数/头 | 必填 | Query head数，必须大于0。 |
| `model_n_kv_head` | 无符号整数/头 | 默认 `model_n_head` | GQA/MQA的KV head数；必须大于0，且 `model_n_head % model_n_kv_head == 0`。 |
| `model_n_embd` | 无符号整数/元素 | 必填 | 隐藏维度；必须能被 `model_n_head` 整除。 |
| `n_tp` | 无符号整数/路 | 必填 | Tensor Parallel并行度。 |
| `n_pp` | 无符号整数/路 | 保留字段 | 部分样例包含，但当前模型初始化没有读取。 |

### 3.5 PIM JSON

示例：`configs/Cases/Nano/Nano_pim_config.json`。

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `dram_type` | 字符串 | 必填 | `newton`或`dram`；当前公开Case使用 `newton`。 |
| `dram_freq` | 无符号整数/MHz | 必填 | DRAM时钟域频率，必须大于0，并与INI `tCK`满足 `dram_freq × tCK ≈ 1000`；启动时按0.1%相对误差强制检查。 |
| `DRAM_act_buf_size_MB` | 无符号整数/MiB | 必填 | 激活数据地址空间预留大小。 |
| `dram_channels` | 无符号整数/通道 | 必填 | 必须大于0、为2的幂，并与INI `[system] channels`一致；启动时会强制检查。 |
| `dram_req_size` | 无符号整数/byte | 必填 | 必须与INI推导的 `bus_width / 8 × BL` 完全一致；启动时会检查，不一致则报错终止。实际DRAM布局仍以INI为权威来源。 |
| `PU_location` | 字符串 | 必填 | 当前样例使用 `bank`；PIM单元按bank组织。 |
| `dual_bank` | 布尔 | 必填 | `true`表示一个PU跨两个bank，要求每通道bank总数为偶数。 |
| `pim_PE_num` | 无符号整数/MAC单元/PU | 必填 | 每个PU内的PE数量，必须大于0。 |
| `pim_input_buffer_size` | 无符号整数/byte | 必填 | 每通道PIM全局输入缓冲大小。 |
| `pim_output_buffer_size` | 无符号整数/byte/PU | 必填 | 每个PU输出缓冲大小。 |
| `pim_parallel_bank_accesses` | 无符号整数/bank | `0` | `0`使用传统COMP地址复制；大于0时仅对PIM_COMP启用bank错开。有效非零值应能被 `dram_channels` 整除，否则回退传统路径。 |
| `hybrid_bonding_bw_area_ratio` | 浮点/mm²/(GB/s) | 必填 | 按总带宽估算hybrid-bonding面积。 |
| `pim_pu_area` | 浮点/mm²/PU | 必填 | 单PU面积。 |
| `pim_controller_area_overhead` | 浮点/mm²/PU | 必填 | 单PU控制器面积开销。 |
| `pim_buffer_area_per_kb` | 浮点/mm²/KiB | 必填 | PIM缓冲面积密度。 |
| `pim_static_power_per_pu` | 浮点/W/PU | 必填 | 单PU静态功耗系数。 |
| `pim_dynamic_power_per_pu_comp` | 浮点/内部能耗系数 | 必填 | 名称保留“power”，当前CA实现按 `系数 × MemoryAccess::size` 累加COMP动态能耗。比较实验必须统一该系数口径。 |
| `pim_buffer_static_power_per_kb` | 浮点/W/KiB | 必填 | 缓冲静态功耗密度。 |
| `pim_buffer_dynamic_power_per_bit` | 浮点/能耗/bit | 必填 | GWRITE/READRES按传输bit数累加的动态能耗系数。 |

## 4. DRAM INI 参数手册

示例：`configs/Cases/Nano/Nano_2xLPDDR5.ini`。INI由PHSim地址分配代码和NewtonSim后端共同读取。所有缺省值来自当前代码，但用于研究结果时应显式填写器件真实参数，避免意外采用通用默认值。

### 4.1 `[system]`

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `channel_size` | 整数/MiB/通道 | `1024` | 必须大于0。会根据器件几何向完整rank容量对齐；过小时会提升到单rank容量。对齐后的rank数必须是2的幂，否则启动失败。 |
| `channels` | 整数/通道 | `1` | 必须大于0、为2的幂，并与PIM JSON `dram_channels`一致；启动时会强制检查。 |
| `bus_width` | 整数/bit/通道 | `64` | 必须大于0、是8的倍数并能被 `device_width` 整除；启动时会强制检查。 |
| `address_mapping` | 12字符字符串 | `chrobabgraco` | 必须恰好由 `ch`、`ra`、`bg`、`ba`、`ro`、`co` 六个二字符字段各组成一次；字符串左侧为高位、右侧为低位。 |
| `queue_structure` | 字符串 | `PER_BANK` | NewtonSim命令队列组织，常用 `PER_BANK`。 |
| `row_buf_policy` | 字符串 | `OPEN_PAGE` | 行缓冲策略，常用 `OPEN_PAGE` 或后端支持的其他值。 |
| `cmd_queue_size` | 整数/命令 | `16` | 每个命令队列容量，必须大于0。 |
| `trans_queue_size` | 整数/事务 | `32` | 事务队列容量，必须大于0。 |
| `unified_queue` | 布尔 | `false` | 是否统一读写事务队列。 |
| `write_buf_size` | 整数/事务 | `16` | 写缓冲容量。 |
| `refresh_policy` | 字符串 | `RANK_LEVEL_STAGGERED` | 仅支持 `RANK_LEVEL_SIMULTANEOUS`、`RANK_LEVEL_STAGGERED`、`BANK_LEVEL_STAGGERED`。 |
| `enable_self_refresh` | 布尔 | `false` | 是否启用self-refresh。 |
| `sref_threshold` | 整数/周期 | `1000` | 进入self-refresh前的空闲阈值。 |
| `aggressive_precharging_enabled` | 布尔 | `false` | 是否启用激进预充电。 |

有效请求粒度：

```text
request_size_bytes = bus_width / 8 × BL
```

地址布局、DataContainer、虚拟页映射和PIM hash都以这个完整burst为统一地址单位。

### 4.2 `[dram_structure]`

| 字段 | 类型/单位 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `protocol` | 字符串 | `DDR3` | 支持 `DDR3`、`DDR4`、`GDDR5`、`GDDR5X`、`GDDR6`、`LPDDR`、`LPDDR3`、`LPDDR4`、`LPDDR5`、`HBM`、`HBM2`、`HMC`。 |
| `bankgroups` | 整数/组 | `2` | bank-group数量，必须是正的2的幂，启动时会强制检查。 |
| `banks_per_group` | 整数/bank/组 | `2` | 每组bank数，必须是正的2的幂，启动时会强制检查。 |
| `bankgroup_enable` | 布尔 | `true` | 关闭时将所有bank合并到一个bank-group。 |
| `rows` | 整数/行/bank | `65536` | 必须是正的2的幂，启动时会强制检查。 |
| `columns` | 整数/物理列/行 | `1024` | 必须是正的2的幂，不小于BL且能被BL整除；启动时会强制检查。 |
| `device_width` | 整数/bit/device | `8` | 必须大于0，且 `bus_width % device_width == 0`；启动时会强制检查。 |
| `BL` | 整数/传输拍 | `8` | Burst length。有效BL必须是正的2的幂。`BL=0`表示理想带宽，内部会按协议默认BL进行容量和地址计算。 |
| `num_dies` | 整数/die | `1` | 堆叠die数量。 |
| `hbm_dual_cmd` | 布尔 | `true` | 仅HBM协议实际启用双命令。 |
| `pim_type` | 字符串 | `SINGLE` | NewtonSim的PIM内存类型；常用 `SINGLE`，dual-bank配置按后端支持值设置。 |

总bank数为 `bankgroups × banks_per_group`；每rank器件数为 `bus_width / device_width`。通道容量会从 `rows × columns × device_width × banks × devices_per_rank` 推导rank数量。上述容量中间量使用64位无符号整数计算；若乘法溢出、单rank不是整MiB，或总地址布局超过64位，仿真器会在启动时报错。外层 `MemConfig` 与Cycle Accurate后端的 `dramsim3::Config` 共用同一几何计算实现，不再分别推导容量和地址位宽。

### 4.3 `[timing]`

除 `tCK` 外，时序参数单位均为 **DRAM周期**。`tCK` 单位为ns。

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `tCK` | `1.0` | DRAM时钟周期，ns；必须为有限正数，并与PIM JSON `dram_freq`满足 `dram_freq × tCK ≈ 1000`。 |
| `AL` | `0` | Additive latency。 |
| `CL` | `12` | CAS read latency。 |
| `CWL` | `12` | CAS write latency。 |
| `tCCD_L` | `6` | 同bank-group列命令间隔。 |
| `tCCD_S` | `4` | 不同bank-group列命令间隔。 |
| `tRTRS` | `2` | Rank-to-rank switching。 |
| `tRTP` | `5` | Read-to-precharge。 |
| `tWTR_L` | `5` | 同bank-group write-to-read。 |
| `tWTR_S` | `5` | 不同bank-group write-to-read。 |
| `tWR` | `10` | Write recovery。 |
| `tRP` | `10` | Precharge latency。 |
| `tRRD_L` | `4` | 同bank-group activate间隔。 |
| `tRRD_S` | `4` | 不同bank-group activate间隔。 |
| `tRAS` | `24` | Activate-to-precharge最小时间。 |
| `tRCD` | `10` | Activate-to-read/write。 |
| `tRFC` | `74` | All-bank refresh时间。 |
| `tCKE` | `6` | CKE最小脉宽。 |
| `tCKESR` | `12` | Self-refresh CKE时间。优先读取规范字段 `tCKESR`；旧字段 `tCKSRE` 可兼容但会警告，两者冲突时终止。 |
| `tXS` | `432` | Self-refresh退出时间。 |
| `tXP` | `8` | Power-down退出时间。 |
| `tRFCb` | `20` | Per-bank refresh时间。 |
| `tREFI` | `7800` | All-bank refresh间隔。 |
| `tREFIb` | `1950` | Per-bank refresh间隔。 |
| `tFAW` | `50` | Four-activate window。 |
| `tRPRE` | `1` | Read preamble。 |
| `tWPRE` | `1` | Write preamble。 |
| `tPPD` | `0` | LPDDR4/GDDR5/6 precharge-to-precharge。 |
| `t32AW` | `330` | GDDR 32-activate window。 |
| `tRCDRD` | `24` | GDDR activate-to-read。 |
| `tRCDWR` | `20` | GDDR activate-to-write。 |
| `ideal_memory_latency` | `10` | 理想内存模式的固定延迟。 |

内部派生关系包括 `tRC=tRAS+tRP`、`RL=AL+CL`、`WL=AL+CWL`。修改协议时应整组替换时序，而不是只修改频率。

仓库部分旧版/协议专用INI还包含下列时序或验证字段，但当前PHSim和所绑定NewtonSim配置解析器**没有读取它们**。它们没有代码默认值，也不会覆盖上表中的通用字段：

| 字段 | 类型/常见单位 | 当前状态与替代字段 |
| --- | --- | --- |
| `activation_window_depth` | 整数/ACT数量 | 旧版激活窗口深度；当前使用固定时序逻辑和 `tFAW`。 |
| `REFRESH_PERIOD` | 整数/周期 | 旧版refresh间隔；当前使用 `tREFI`/`tREFIb`。 |
| `tCAS` | 整数/周期 | 旧版CAS延迟；当前分别使用 `CL`、`CWL`。 |
| `tCMD` | 整数/周期 | 旧版命令总线延迟；当前未建模为独立参数。 |
| `tCWD` | 整数/周期 | 旧版写数据延迟；当前使用 `CWL`。 |
| `tPDD` | 整数/周期 | LPDDR专用旧字段；当前未读取。 |
| `tRFC2`、`tRFC4` | 整数/周期 | Fine-granularity refresh旧字段；当前使用 `tRFC`。 |
| `tRFCPB` | 整数/周期 | Per-bank refresh旧字段；当前使用 `tRFCb`。 |
| `tRREFD` | 整数/周期 | Refresh-to-refresh delay旧字段；当前未读取。 |
| `tRTP_L`、`tRTP_S` | 整数/周期 | 分bank-group read-to-precharge旧字段；当前只读取通用 `tRTP`，缺失时默认5。 |
| `tWR2` | 整数/周期 | 旧版第二write-recovery参数；当前只读取 `tWR`。 |
| `validation_output` | 字符串/路径 | 旧验证器输出文件名；当前PHSim不生成该文件。 |

这些字段可以为了保留原始器件配置而继续存在，但实验报告不能把它们列为“当前生效参数”。

### 4.4 `[power]`

| 字段 | 类型/单位 | 默认值 | 说明 |
| --- | --- | ---: | --- |
| `VDD` | 浮点/V | `1.2` | 供电电压。 |
| `IDD0` | 浮点/mA | `48` | ACT-PRE工作电流。 |
| `IDD2P` | 浮点/mA | `25` | Precharge power-down电流。 |
| `IDD2N` | 浮点/mA | `34` | Precharge standby电流。 |
| `IDD3N` | 浮点/mA | `43` | Active standby电流。 |
| `IDD4W` | 浮点/mA | `123` | Burst write电流。 |
| `IDD4R` | 浮点/mA | `135` | Burst read电流。 |
| `IDD5AB` | 浮点/mA | `250` | All-bank refresh电流。 |
| `IDD5PB` | 浮点/mA | `5` | Per-bank refresh电流。 |
| `IDD6x` | 浮点/mA | `31` | Self-refresh电流。 |
| `IDD3P` | 浮点/mA | 保留字段 | 某些INI中存在，但当前功耗初始化行被注释，不参与计算。 |

部分ST/DDR器件INI还保留 `IDD1`、`IDD2Q`、`IDD3Pf`、`IDD3Ps`、`IDD6L`、`IDD7` 和 `IPP0`。它们都是浮点电流参数（通常为mA），当前功耗解析器不读取，也没有代码默认值；当前能耗模型只使用上表明确列出的 `IDD0/2P/2N/3N/4W/4R/5AB/5PB/6x`。

DRAM命令能耗按 `V × mA × ns` 计算，因此JSON/TXT中的DRAM能耗数值单位为pJ。每个rank还会乘以 `devices_per_rank`。PIM能耗使用PIM JSON中的经验系数；只有在这些系数采用一致物理单位时，DRAM与PIM总能耗才可直接作绝对值解释。

### 4.5 `[other]`、`[hmc]` 与 `[thermal]`

| section/字段 | 类型/单位 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `other.epoch_period` | 整数/DRAM周期 | `100000` | NewtonSim epoch统计周期。 |
| `other.output_level` | 整数 | `1` | `0`仅summary，`1`增加epoch输出，`2`增加直方图；`-1`在代码注释中标为未完整实现。 |
| `other.output_prefix` | 字符串 | `dramsim3` | CA输出文件前缀。PHSim ED固定使用 `eventdrivendram`。 |
| `hmc.num_links` | 整数/条 | `4` | HMC专用链路数。 |
| `hmc.link_width` | 整数/bit | `16` | HMC链路宽度。 |
| `hmc.link_speed` | 整数/MHz | `15000` | HMC链路速率。 |
| `hmc.block_size` | 整数/byte | `64` | HMC块大小，必须为32 byte的倍数。 |
| `hmc.xbar_queue_depth` | 整数/事务 | `16` | HMC crossbar队列深度。 |

部分通用INI还包含 `[thermal]`：`loc_mapping`（地址到物理位置映射字符串）、`power_epoch_period`（周期）、`chip_dim_x`/`chip_dim_y`（m）、`amb_temp`（摄氏度）、`mat_dim_x`/`mat_dim_y`（网格数）、`bank_order`/`bank_layer_order`（布局枚举）、`logic_bg_power`/`logic_max_power`（逻辑层功耗系数）。默认PHSim Release构建没有启用 `THERMAL`，这些字段不影响当前结果、没有当前构建的代码默认值；只有自行打开并完整接入thermal模块后才使用。

## 5. 输出文件与指标

### 5.1 稳定输出文件

| 文件 | 生成条件 | 内容 |
| --- | --- | --- |
| `_summary.tsv` | 所有正常结束的运行 | 按stage汇总的核心周期、PIM周期和DRAM带宽利用率。 |
| `core_timing.tsv` | 所有正常结束的运行 | 每个核心的计算、访存停顿、空闲和算子级周期。 |
| `icnt_traffic.json` | 所有正常结束的运行 | 每通道实测、估计和逻辑流量/PIM请求数。 |
| `dramsim3.json`、`dramsim3.txt` | CycleAccurate | NewtonSim逐通道命令、延迟、状态周期、带宽和能耗。前缀可由INI修改。 |
| `eventdrivendram.json`、`eventdrivendram.txt` | EventDriven | ED逐通道请求、合并、命令、逻辑补偿、状态周期、带宽和能耗。 |
| `proportional_command_compensation.json` | CycleAccurate | Proportional采样的CA命令补偿；关闭采样时可能为空。 |
| `attention_core_round_commands.tsv` | 相关注意力采样路径 | 每轮注意力采样/命令记录，仅特定加速case产生。 |
| `log.txt` | 所有运行 | 当前只由独立文件logger写入结束信息；大部分spdlog统计输出仍在终端。建议用shell重定向保存完整控制台。 |

测试脚本生成的 `test-process.log` 是CTest封装捕获的终端输出，不是直接运行可执行程序时自动生成的文件。

保存终端输出示例：

```bash
./NMC_Simulator -sc ../tests/configs/data_container_gemm_ca_simulation.json \
  -o output/gemm-ca 2>&1 | tee output/gemm-ca/process.log
```

### 5.2 `_summary.tsv`

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `Stage` | 枚举名称 | `Prefill`、`Decode`、`Single_test`、`Multi_test`等完成stage。 |
| `total_cycles` | 核心周期 | 当前stage完成核心周期减去上一stage完成核心周期。 |
| `pim_cycles` | DRAM周期/通道平均 | 当前stage中PIM命令占用周期的每通道平均值；采样开启时包含逻辑补偿。 |
| `mem_bw_util` | 百分数值 | 后端报告的平均DRAM带宽利用率，例如 `42.5` 表示约42.5%。 |

`total_cycles` 是仿真时间轴上的核心周期，不是主机真实运行时间。主机组件耗时只打印在终端的 `Component Real Time Breakdown` 中。

### 5.3 `core_timing.tsv`

| 字段族 | 单位 | 含义 |
| --- | --- | --- |
| `core` | ID | 核心编号。 |
| `total_cycle` | 核心周期 | 该核心最终本地时钟。 |
| `accounted_cycle` | 核心周期 | `compute_cycle + active_memory_stall + idle_cycle`的统计口径；代码中由compute和memory统计构成。 |
| `compute_cycle` | 核心周期 | 逻辑计算活动周期，包含已执行和已补偿工作。 |
| `active_memory_stall` | 核心周期 | `memory_cycle - idle_cycle`，下限为0。 |
| `idle_cycle` | 核心周期 | 没有可执行工作造成的空闲周期。 |
| `estimated_compute` | 核心周期 | 因采样跳过而估计补回的计算周期。 |
| `estimated_memory` | 核心周期 | 因采样跳过而估计补回的访存周期。 |
| `estimated_idle` | 核心周期 | 因采样跳过而估计补回的空闲周期。 |
| `compute_util` | `[0,1]` | `compute_cycle / accounted_cycle`。 |
| `memory_stall_util` | `[0,1]` | `active_memory_stall / accounted_cycle`。 |
| `idle_util` | `[0,1]` | `idle_cycle / accounted_cycle`。 |
| `*_cycle` | 核心周期 | GEMM、LayerNorm、RMSNorm、RoPE、Softmax、Add、Mul、GELU、SiLU、GEMV各自计算周期。 |
| `*_stall` | 核心周期 | 对应算子等待存储系统的周期。 |

### 5.4 `icnt_traffic.json`

顶层key是通道编号。每个通道包含以下命名规则：

- `measured_*`：仿真器实际执行并通过互连的工作。
- `estimated_*`：采样跳过后由补偿逻辑估计的工作。
- `logical_* = measured_* + estimated_*`：用于不同加速设置之间比较的完整逻辑工作量。

流量字段 `read_bytes`、`write_bytes` 单位为byte。PIM字段 `pheader_requests`、`gwrite_requests`、`comp_requests`、`readres_requests` 单位为请求条数。`COMP_HASH`计入 `comp_requests`；组合COMP+READRES会同时计入两类。

### 5.5 DRAM JSON/TXT公共指标

#### 完成量与命令计数

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `num_reads_done`、`num_writes_done` | 事务 | 完成的普通DRAM读/写。 |
| `num_pim_done` | 事务 | 完成的PIM请求；ED还提供measured/estimated/logical拆分。 |
| `num_read_cmds`、`num_write_cmds` | 命令 | 发出的DRAM列读/写命令。一次逻辑事务不一定对应一个命令。 |
| `num_act_cmds`、`num_pre_cmds` | 命令 | ACT和PRE命令。 |
| `num_ref_cmds`、`num_refb_cmds` | 命令 | All-bank和per-bank refresh命令。 |
| `num_pheader_cmds` | 命令 | PIM header/准备命令。 |
| `num_gwrite_cmds` | 命令 | PIM全局输入缓冲写命令。 |
| `num_comp_cmds` | 命令 | PIM计算命令，包括hash计算路径。 |
| `num_readres_cmds` | 命令 | PIM结果读取命令。 |
| `num_pim_cmds` | 命令 | PIM命令总口径。 |
| `num_pim_activate_cmds`、`num_pim_precharge_cmds` | 命令 | PIM相关行激活和预充电。 |
| `num_read_row_hits`、`num_write_row_hits` | 事务 | 读/写行命中。 |
| `num_write_buf_hits` | 事务 | 写缓冲命中/合并。 |
| `num_ondemand_pres` | 命令 | 因目标行冲突触发的按需预充电。 |

ED还输出 `num_read_merges`、`num_write_merges` 和 `num_*_requests`，用于区分上层请求、合并后的物理完成和底层命令。不要把request、transaction和command三个层次混为一谈。

#### 周期、延迟与带宽

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `num_cycles` | DRAM周期 | 后端实际推进的时钟。 |
| `estimated_cycles` | DRAM周期 | 采样补偿周期，仅ED显式单列。 |
| `logical_cycles` | DRAM周期 | ED中完整逻辑周期口径。 |
| `pim_cycles` | DRAM周期 | PIM命令占用周期。ED另有estimated/logical字段。 |
| `average_read_latency` / `read_latency` | DRAM周期 | 已完成读事务平均延迟/累计延迟及直方图。 |
| `write_latency` | DRAM周期 | 写延迟累计值及直方图。 |
| `average_interarrival` | DRAM周期 | 事务到达间隔平均值。 |
| `avg_memory_bandwidth` | GB/s口径 | 后端按完成数据量和仿真时间计算的平均带宽。 |
| `average_power` | 内部功率口径 | `total_energy / logical_cycles`；需要结合 `tCK` 和能耗单位解释，不等同于直接测量的W。 |
| `rank_active_cycles`、`all_bank_idle_cycles` | rank×周期数组 | 每个rank至少一行激活、或全部bank空闲的周期。 |
| `pim_rank_active_cycles`、`pim_all_bank_idle_cycles` | rank×周期数组 | PIM状态对应的活跃/空闲周期。 |

CA和ED采用不同的调度抽象，命令发出时刻、write-buffer合并和行状态细节可能不同。正确的一致性检查顺序是：

1. stage是否完成、是否存在未归还请求；
2. `icnt_traffic.json` 的 `logical_read_bytes`、`logical_write_bytes` 和逻辑PIM请求；
3. DRAM逻辑读写/PIM完成量；
4. 最后比较周期、行命中、ACT/PRE/WRITE等后端敏感计数。

#### 能耗

| 字段 | 含义 |
| --- | --- |
| `act_energy` | ACT动态能耗。 |
| `read_energy`、`write_energy` | 普通DRAM读写动态能耗。 |
| `ref_energy`、`refb_energy` | All-bank/per-bank refresh能耗。 |
| `background_energy` | rank active/idle背景能耗。 |
| `gwrite_energy`、`comp_energy`、`readres_energy` | PIM输入写、计算、结果读动态能耗。 |
| `pim_background_energy` | PU和PIM缓冲静态背景能耗。 |
| `pim_dynamic_energy` | GWRITE+COMP+READRES动态能耗。 |
| `total_pim_energy` | PIM背景能耗+PIM动态能耗。 |
| `total_energy` | DRAM动态、refresh、背景和PIM能耗总和。 |

当前模型没有为PIM PRE单独增加动态能耗项；即使 `num_pim_precharge_cmds` 非零，也不应自行从命令数推断一个未实现的PIM PRE能耗。

### 5.6 主机真实时间

终端的 `Component Real Time Breakdown` 统计的是仿真器在主机上消耗的wall-clock时间：

- `program compile time`、`deferred compile time`：算子图/tile构造；
- `client+scheduler time`：请求与调度；
- `core time`：核心推进；
- `dram+icnt time`：DRAM和互连推进；
- `runtime excl compile`：组件计时总和，不含编译；
- `component total incl compile`：上述总和加编译。

它们不能与 `_summary.tsv` 的仿真周期直接相加或互换。

## 6. 典型使用案例

所有命令都假设已经构建，并从 `build/` 目录运行。每个case使用独立输出目录，避免上一次结果混入。

### 6.1 完整模型路径（建议先做单层prefill）

完整模型路径要求：

```json
{
  "test_single_op": false,
  "test_multi_layer": false,
  "gen_request": true,
  "gen_request_count": 1,
  "gen_request_input_size": 8,
  "gen_random_request": false,
  "accelerate_ctrl": false
}
```

不要只保存上述片段；PHSim不支持JSON overlay。应复制完整的 `configs/Cases/inference_config.json`，修改这些字段，再复制顶层simulation JSON并把 `inference_config_file_path` 指向新文件。

建议第一轮使用：

- Nano硬件；
- `opt-350M.json`；
- `model_n_layer=1`；
- 输入8或16 token；
- CA/ED任选其一；
- DataContainer和虚拟内存先关闭。

运行后确认 `_summary.tsv` 中出现完整模型stage而非 `Single_test`/`Multi_test`，并检查 `core_timing.tsv`、ICNT和DRAM输出。默认 `output_token_iteration_enable=false`，因此保持原有固定的 `Prefill → Decode → NPU_Decode` stage序列；此模式不会按照 `gen_request_output_size` 重复Decode。

### 6.2 按输出token数迭代Decode

要让 `gen_request_output_size` 真正控制Decode轮数，使用完整模型模式并显式开启：

```json
{
  "test_single_op": false,
  "test_multi_layer": false,
  "allocation_scheme": "DASH",
  "gen_request": true,
  "gen_request_count": 1,
  "gen_request_input_size": 4,
  "gen_request_output_size": 2,
  "output_token_iteration_enable": true,
  "gen_random_request": false,
  "max_seq_len": 16
}
```

上述 `DASH` 配置的预期stage为 `Prefill → Decode → Decode`；日志依次出现 `generated token 1/2` 和 `generated token 2/2`。若把 `allocation_scheme` 改成 `NPU`，对应stage为 `Prefill → NPU_Decode → NPU_Decode`。

仓库内可直接参考并由CTest覆盖的完整小型配置：

- `tests/fixtures/smoke/simulation_iterative_decode_cycle_accurate.json`；
- `tests/fixtures/smoke/simulation_iterative_decode_event_driven.json`。

PIM Decode不能任意缩小模型：当前权重分块会按 `column_interleave × total_banks` 取整，输出维度过小时会得到零长度PIM tile；VCache的最小分块也必须放入 `pim_input_buffer_size`。回归case使用 `model_n_embd=256` 和 `pim_input_buffer_size=4096 byte`，规模仍较小，同时满足当前PIM分块约束。

### 6.3 单算子GEMM

仓库提供经过CTest验证的小型配置：

```bash
mkdir -p output/gemm-ca
./NMC_Simulator \
  -sc ../tests/configs/data_container_gemm_ca_simulation.json \
  -o output/gemm-ca
```

该case设置：`test_single_op=true`、`test_single_op_name=gemm`、CA、DataContainer开启、虚拟内存关闭。预期 `_summary.tsv` 的stage为 `Single_test`，并产生普通DRAM读写命令。

若只想测时序而不维护数据，把对应inference JSON中的 `dram_data_container_enable` 改为 `false`。

### 6.4 CA与ED切换和对比

唯一后端开关是：

```json
"dram_trace_simulation_mode": false
```

- `false`：CycleAccurate/NewtonSim，输出 `dramsim3.*`；
- `true`：EventDriven，输出 `eventdrivendram.*`。

对同一case做对比时，只修改这个布尔值并使用不同输出目录。仓库已有GEMM+DataContainer配对配置：

```bash
mkdir -p output/gemm-ca output/gemm-ed

./NMC_Simulator \
  -sc ../tests/configs/data_container_gemm_ca_simulation.json \
  -o output/gemm-ca

./NMC_Simulator \
  -sc ../tests/configs/data_container_gemm_ed_simulation.json \
  -o output/gemm-ed
```

至少比较：stage完成量、`logical_read_bytes`、`logical_write_bytes`、读写/PIM完成量和最终周期。不要要求CA/ED的WRITE发出数、ACT/PRE数或行命中数天然完全相同。

### 6.5 DataContainer

最小设置：

```json
{
  "dram_data_container_enable": true,
  "dram_data_container_max_payload_mb": 64
}
```

DataContainer只为访问过的DRAM列物化存储；未写位置读为0。它在CA和ED的读完成、带数据写完成路径中维护burst payload。普通统计文件不会导出DataContainer内容，因此功能验证应运行专用测试：

```bash
ctest --test-dir . --output-on-failure \
  -R "data_container|memory_access_lifecycle_gemm"
```

大模型启用真实数据维护时，先用1请求、8 token、单算子和较小payload上限。统计结果一致只说明逻辑工作量一致；DataContainer正确性由专用读回/写回测试判断。

### 6.6 虚拟内存与地址hash

只验证地址映射/PIM hash、不开DataContainer：

```bash
mkdir -p output/hash-ca output/hash-ed

./NMC_Simulator \
  -sc ../tests/configs/pim_hash_smoke_ca_simulation.json \
  -o output/hash-ca

./NMC_Simulator \
  -sc ../tests/configs/pim_hash_smoke_ed_simulation.json \
  -o output/hash-ed
```

同时验证虚拟内存、PIM COMP hash和DataContainer：

```bash
mkdir -p output/vm-dc-ca output/vm-dc-ed

./NMC_Simulator \
  -sc ../tests/configs/data_container_vm_ca_simulation.json \
  -o output/vm-dc-ca

./NMC_Simulator \
  -sc ../tests/configs/data_container_vm_ed_simulation.json \
  -o output/vm-dc-ed
```

对应关键开关为：

```json
{
  "allocation_scheme": "DASH",
  "virtual_mem_hash_enable": true,
  "dram_data_container_enable": true,
  "test_single_op": true,
  "test_single_op_name": "pim_gemv"
}
```

验证重点：

- 两个后端都完成且无outstanding request；
- ICNT逻辑PIM请求量一致；
- DRAM输出包含COMP/PIM相关完成与命令；
- 专用 `virtual_memory_address_layout_test` 和 `pim_hash_address_group_test` 通过；
- 开关关闭后的基线case保持原有周期和流量口径。

## 7. 实验检查清单

运行前：

1. 输出目录已创建且为空或使用新名称；
2. 所有路径相对于 `build/` 有效；
3. PIM `dram_channels` 等于INI `channels`；
4. DRAM几何是2的幂且 `bus_width/device_width/BL`一致；
5. `model_n_embd % model_n_head == 0`；
6. `model_n_head % model_n_kv_head == 0`；
7. 基线对比关闭所有采样；
8. DataContainer先设置payload上限并使用小case；
9. 开启输出token迭代时确认 `input_size + output_size <= max_seq_len`，并确认PIM分块不为0。

运行后：

1. 进程返回0且出现 `Finish the simulation`；
2. 没有outstanding `MemoryAccess`异常；
3. `_summary.tsv` stage符合预期；
4. `core_timing.tsv` 中利用率有限且非NaN；
5. ICNT `logical = measured + estimated`；
6. DRAM完成量与ICNT请求/流量口径匹配；
7. CA/ED对比记录配置差异，而不是只保存结果文件；
8. 能耗比较使用相同INI/PIM系数和相同逻辑工作量。

## 8. 已知配置兼容性提示

- `gen_request_output_size` 已用于固定请求的输出目标，但只有 `output_token_iteration_enable=true` 才按该目标重复Decode；默认关闭时仍执行原有固定stage序列。
- `gen_request=false` 时，当前Client仍用 `gen_request_output_size` 作为输出目标；请求trace解析得到的答案长度尚未传入请求对象。
- PIM Decode的维度必须满足当前bank/column交织粒度；过小的隐藏维度或过小的 `pim_input_buffer_size` 可能产生零长度分块，优先从已验证的迭代smoke配置缩放。
- `dram_req_size`不是DRAM布局的权威控制量，但必须与INI的 `bus_width / 8 × BL` 一致；`sram_size`、`process_bit`、`n_pp`等样例字段不是当前对应子系统的权威控制量。
- 旧字段 `tCKSRE` 仅用于兼容：单独出现时作为 `tCKESR` 读取并警告；两者同时出现但数值冲突时终止；均缺失时使用12周期默认值。
- `record_dram_completion_trace` 还受编译期 `ENABLE_DRAM_ALIGNMENT_TRACE` 控制，默认Release仅设置JSON不会生成trace。
- 顶层Case引用的请求trace在 `gen_request=true` 时不会被读取；切换到 `gen_request=false` 前必须确认文件真实存在。
- 仓库模型配置为了控制仿真规模均默认 `model_n_layer=1`；扩大层数前先评估运行时间和主机内存。

遇到结果差异时，请保存：Git提交号、完整六层配置、启动命令、终端日志和整个输出目录。只有这些条件固定后，周期、流量和能耗结果才具有可复现性。
