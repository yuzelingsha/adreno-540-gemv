# Adreno 540 GEMV / AdrenoLLM

面向 **Xiaomi MIX 2 / Snapdragon 835 / Adreno 540** 的 OpenCL INT4 GEMV 优化项目，通过动态 Hook 在 Google LiteRT-LM 运行时中替换内核源码与工作组布局，研究 MiniCPM 单 Token 自回归解码的性能和数值一致性。

**现存实测归档中的最高调优均值为 19.94 ± 0.05 tok/s，对应约 50.16 ms/token；异步 q-VRL 配置为 18.67 ± 0.08 tok/s。** 两组记录均为五轮、每轮 512 个输出 Token 的解码统计。19.94 是特定锁频与异步配置下的历史测试成绩，初始化、提示词处理和网页传输耗时另行计量。

本 README 按现存 `result.json`、环境配置、轮次汇总日志和源码整理。性能数字属于历史实验记录，本次文档更新未重新运行设备。下文区分归档结果、日志重算结果与尚缺完整证据的项目。

## 1. 测试设备、模型与运行时

| 项目 | 归档配置 |
|:---|:---|
| 设备 | Xiaomi MIX 2，代号 `chiron` |
| SoC / GPU | Qualcomm Snapdragon 835（MSM8998）/ Adreno 540 |
| GPU 频率 | 710 MHz |
| 操作系统 | Android 9 / LineageOS 16.0 |
| 推理运行时 | Google LiteRT-LM，实验记录的提交为 `84263b0` |
| OpenCL 驱动 | Qualcomm OpenCL 2.0，归档标注为 2019 年版本 |
| 模型归档名称 | MiniCPM-1B-Q4_0；早期文档另称 MiniCPM5-1B |
| 量化格式 | Transformer 矩阵 INT4 block32、FP16 scales；LM-Head INT8 |
| LM-Head 形状 | `[130560, 1536]` |
| 脚本使用的模型路径 | `/data/local/tmp/minicpm_cpuprep.litertlm` |

当前公开归档没有模型文件与运行时二进制的完整哈希。复测时应保存实际文件哈希、驱动版本和设备状态，确认所比较的模型及运行时一致。

## 2. 端到端解码结果

### 2.1 异步与锁频调优

| 配置 | 归档均值 ± 标准差（tok/s） | 平均步时延（ms/token） | 五轮速度范围（tok/s） | 结果文件 |
|:---|:---:|:---:|:---:|:---|
| q-VRL Async | **18.67 ± 0.08** | **53.56** | 18.55–18.78 | [q_vrl_async/result.json](experiments/tuned/q_vrl_async/result.json) |
| Gold Peak Lock | **19.94 ± 0.05** | **50.16** | 19.88–20.01 | [gold_lock/result.json](experiments/tuned/gold_lock/result.json) |

平均步时延来自归档汇总，是解码吞吐的倒数；现有记录没有逐步时延分布，不能据此给出 P95 时延或首 Token 时延。

两组本地保留的 `stdout.log` 轮次汇总如下。耗时为记录中各轮生成 512 个 Token 的解码耗时，初始化时间在日志中单独列出。

| 轮次 | Async 解码耗时（s） | Async 吞吐（tok/s） | Gold 解码耗时（s） | Gold 吞吐（tok/s） |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 27.59 | 18.55 | 25.75 | 19.88 |
| 2 | 27.42 | 18.67 | 25.68 | 19.94 |
| 3 | 27.26 | 18.78 | 25.59 | 20.01 |
| 4 | 27.45 | 18.65 | 25.69 | 19.93 |
| 5 | 27.38 | 18.70 | 25.67 | 19.95 |

按表中已四舍五入的吞吐值重算，Async 为 `18.670 ± 0.083`、Gold 为 `19.942 ± 0.047 tok/s`，标准差采用样本标准差（`N−1`）。因此 **19.94 是五轮均值，20.01 是五轮中最高的单轮平均速度**。现有轮次记录没有支撑“20.00 tok/s 瞬时单步峰值”的逐 Token 时间戳。

配置来源为 [Async 环境文件](experiments/tuned/q_vrl_async/environment.yaml)和 [Gold 环境文件](experiments/tuned/gold_lock/environment.yaml)：

| 参数 | q-VRL Async | Gold Peak Lock |
|:---|:---|:---|
| `LITERT_GPU_WAIT_FOR_COMPLETION` | `0` | `0` |
| `LITERT_GPU_KERNEL_BATCH_SIZE` | `12` | `12` |
| CPU affinity mask | `0x0c` | `0x0c` |
| GPU 频率 | 710 MHz | 710 MHz，`performance` |
| CPU 频率 | 环境文件未记录具体频点 | 环境文件记录 2.4576 GHz，结果文件简写为 2.45 GHz |
| `thermal-engine` | 记录为 stopped | 记录为 stopped；轮次汇总标注环境温度 29°C |

这里的 `kernel_batch_size=12` 是驱动内核提交批量，模型解码仍为 Batch=1。掩码 `0x0c` 表示 CPU 2、3，不等于大核编号；Gold 文件同时写了 Gold cluster 与该掩码，其他文档又写 2.36 GHz。这些硬件描述存在冲突，复测应读取实际 cpufreq policy、在线核心与进程 affinity，不能从掩码或文档名称推定频率。

### 2.2 参考组与失败候选

下表保留各 `result.json` 的原有汇总字段，并列出本地五轮汇总日志的重算值。重算只使用日志中已保留的吞吐数字，没有补造精度更高的测量值。

| 方案 | JSON 归档均值 ± 标准差（tok/s） | 五轮日志重算均值 ± 样本标准差（tok/s） | 归档正确性结论 | 结果文件 |
|:---|:---:|:---:|:---|:---|
| Baseline | 10.98 ± 0.11 | 10.98 ± 0.12 | 参考基线 | [baseline/result.json](experiments/reproducible/baseline/result.json) |
| blockscale | 11.85 ± 0.14 | 11.84 ± 0.14 | 未记录发散 | [blockscale/result.json](experiments/reproducible/blockscale/result.json) |
| q-VRL | 14.32 ± 0.18 | 14.30 ± 0.16 | 未记录发散 | [q_vrl/result.json](experiments/reproducible/q_vrl/result.json) |
| add-y4 | 13.78 ± 0.22 | 13.75 ± 0.22 | 第 187 步发散，弃用 | [add_y4/result.json](experiments/final_validation/add_y4/result.json) |
| add-y4 + q-VRL | 16.98 ± 0.25 | 16.95 ± 0.22 | 第 187 步发散，弃用 | [add_y4_q_vrl/result.json](experiments/tuned/add_y4_q_vrl/result.json) |

这些目录的 README、JSON 和脚本没有完整统一硬件与等待配置。例如，参考组日志标题称默认 CFS 调度，但 JSON 又记录了 CPU 2.45 GHz 与温控关闭。因此它们保留为历史对照资料；现有文件不足以支持“无锁频默认环境下的纯 q-VRL 净收益”或跨组严格同条件加速比。

旧 README 中的 **16.82 tok/s** 属于早期 64 Token 测试的汇总口径；当前调优归档扩展到了 512 Token。早期 16.x、参考组 14.x 和调优组 18–19.x 应连同各自的配置和证据阅读。

### 2.3 数值正确性的证据范围

q-VRL、Async 与 Gold 的结果文件记录 `fp16_bit_exact=true` 和 `first_divergence_step=null`；[长生成结果](experiments/final_validation/long_generation/result.json)也记录 q-VRL 在 64、256、512 Token 下匹配。

公开的 `tokens.txt` 含省略号，缺少对应测试的完整 512 步基线与候选序列。现有速度日志属于轮次汇总，完整 LiteRT trace 与逐 Token 时间戳未随仓库发布。因此 README 将上述正确性结论表述为**归档报告的结果**。完整 Token 序列匹配证明受测输入的生成轨迹一致；FP16 张量逐位一致还需独立张量比对，不能从 Token ID 一致推广到任意输入的全部中间结果。

`*.log` 和 `*raw*.txt` 受 `.gitignore` 规则排除，克隆仓库不会自动取得本地的完整日志。本节已列出现存调优轮次汇总，公开数值入口为各目录的 `result.json`。

## 3. 实现与适用范围

[DLSYM Hook](hook/adrenollm_dlsym_hook.cpp)拦截 OpenCL 源码编译和内核提交。`ADRENOLLM_DLSYM_BLOCK_SCALE` 控制 block-scale 重写；`ADRENOLLM_DLSYM_Q_VIRTUAL_Y4` 控制非 Add INT4 模板的替换，并对特定 dispatch 几何调整工作组。

当前 q-VRL dispatch 条件为 `global=(512,16,…)`、`local=(4,16,…)`，重映射为 `global=(512,4,…)`、`local=(16,4,…)`。识别依赖内核源码模板和工作组尺寸，代码没有通过 Transformer 算子名称建立通用绑定，适用范围依赖受测模型及 LiteRT 生成的内核。

[虚拟车道候选内核](kernels/vrl/plain_virtual_y4.cl)使用四组寄存器累加保留 16 个逻辑车道，并显式执行 `8 → 4 → 2 → 1` 规约树。该设计以保留逻辑规约拓扑为目标，实际数值结果仍需对应源码、缓存和编译条件下的验证。现有源码与性能归档没有提供 LMS bank-conflict 硬件计数器，性能改善的微架构归因仍需 profiling 支持。

Hook 还包含 LM-Head 局部激活暂存、零点处理、展开、add-y4 和静态回放等研究开关。源码中存在某个开关不代表它已经通过精度或性能验证；生产候选仍以 q-VRL 及其匹配的缓存为主，失败候选应保留独立配置。

## 4. LM-Head 微基准与带宽口径

LM-Head 的 INT8 权重数据面为 `130560 × 1536 = 200,540,160` 字节，即十进制 **200.54 MB**；隐藏向量 FP16 容量为 3072 字节，130,560 个 FP16 logits 的逻辑容量为 261,120 字节。200.54 MB 描述词表输出矩阵，完整模型还包含 Transformer 权重和其他数据。

[LM-Head 微基准](benchmarks/lmhead_microbench/)单独拆解权重读取、解包、反量化、点积与局部激活暂存。[单 Token 报告](docs/a540_minicpm_single_token_report.md)记录了以下算子测试值：

| 测试阶段 | 文档归档耗时 |
|:---|:---:|
| Stage A：权重读取 | 17.68 ms |
| Stage B：权重读取与解包 | 17.62 ms |
| Stage C：反量化 | 17.09 ms |
| Stage D：全局激活读取与点积 | 28.73 ms |
| Stage D Opt 2：工作组局部激活暂存 | 17.21 ms；记录最小值 16.81 ms |

这些值属于独立微基准报告，当前仓库未提供相应完整运行日志，也没有据此证明端到端每个 Token 固定节省 11.52 ms。微基准实现会按工作组计入局部暂存所需的全局激活读取；该优化减少重复读取，不能写成“片外激活流量清零”。

Batch=1 GEMV 权重复用少，内存读取、内核提交、采样与 KV cache 共同影响解码耗时。估算带宽必须使用对应算子的字节量与耗时；完整解码的物理 DRAM 流量还受缓存命中、驱动布局与上下文长度影响。现有资料没有充分支撑“系统可用带宽固定为某个值”“总线利用率已达 90%–94%”或“20 tok/s 是不可跨越的硬上限”，这些推导不作为本 README 的实测结论。

## 5. 构建与复测入口

### 5.1 准备依赖

目标设备需要 Root、ADB、匹配的 AArch64 Android LiteRT-LM 可执行文件及动态库、模型文件、Hook 与配套 OpenCL 内核。运行时、模型权重及设备 JIT 缓存未随仓库分发。

Hook 的 [Makefile](hook/Makefile)提供以下交叉编译参数：

```bash
make -C hook NDK=/path/to/android-ndk OPENCL_DIR=/path/to/opencl API=28
```

将路径替换为本机 Android NDK 与 OpenCL 头文件目录。目标产物为 AArch64 Android `libadrenollm_dlsym_hook.so`；当前 Makefile 在找不到 NDK 编译器时会尝试宿主 `clang++`，复测部署前需核对产物架构和依赖。

### 5.2 核对手机端文件与缓存

现有脚本依赖下列手机端资产：

| 路径 | 用途 |
|:---|:---|
| `/data/local/tmp/litert-a540/litert_lm_advanced_main` | LiteRT-LM 执行程序 |
| `/data/local/tmp/litert-a540/` | 匹配的运行时动态库与 Hook；按实际依赖准备 `libGemmaModelConstraintProvider.so` 等库 |
| `/data/local/tmp/minicpm_cpuprep.litertlm` | 模型权重 |
| `/data/local/tmp/llm-safe` | 脚本调用的执行包装器，仓库未提供其源码 |
| `/data/local/tmp/litert-a540/plain_block_dual.cl` | q-VRL Hook 编译期读取的 dual 内核 |
| `/data/local/tmp/program_cache.blockscale1413.bin` | blockscale 对照缓存 |
| `/data/local/tmp/program_cache.qvrl_only.bin` | q-VRL 对照缓存 |

**目前存在一个公开复现缺口：Hook 读取 `plain_block_dual.cl`，仓库提供的是 `kernels/vrl/plain_virtual_y4.cl`，缺少同名 dual 文件及其部署映射。** 现有手机上的预置文件或缓存可能满足该依赖，但从仓库重新编译和冷启动时仍需补齐匹配资产。不能仅凭重命名候选文件认定它与设备缓存中的内核等价。

OpenCL 程序缓存命中时会绕过源码编译 Hook。切换优化开关前应明确缓存对应的源码版本；仅修改环境变量不能证明运行了对应候选。

### 5.3 运行已有对照脚本

完成上述准备后，[干净 A/B 脚本](scripts/run_qvrl_clean_ab.sh)是现有对照入口：

```bash
adb push scripts/run_qvrl_clean_ab.sh /data/local/tmp/
adb shell "su -c 'sh /data/local/tmp/run_qvrl_clean_ab.sh'"
```

该脚本按 `BASE_A → Q_A → Q_B → BASE_B` 顺序运行，每轮 prefill 16、decode 64 Token；显式清除 add-y4 / ADD_VIRTUAL / LOCAL_SRC384 候选，使用 `Wait=1`、`Batch=12`、mask `0x0c`，并在两种预置缓存间切换。它输出初始化和解码统计，不采集完整 Token ID，也不自动复现 Gold 五轮锁频测试。

[test_speed.sh](scripts/test_speed.sh)使用异步等待、Batch=12、256 个 decode Token；[test_speed_20.sh](scripts/test_speed_20.sh)使用异步等待、Batch=16、512 个 decode Token。两个脚本都设置 mask `0xf0`，并未完整设置 Gold 的锁频条件或显式清除继承的所有研究开关，因此运行前需核对实际环境。

`ADRENOLLM_FORCE_GPU_SAMPLER=1` 与命令行 `--sampler_backend=cpu` 同时出现在这些脚本中，最终生效的采样路径依赖外部运行时与 Hook 组合，应从实际运行日志确认。

### 5.4 保存性能与正确性证据

复测需要保存运行时/模型/Hook/内核/缓存哈希、实际等待与 batch 参数、CPU affinity、cpufreq policy、GPU 频率、温度，以及每轮完整启动和退出日志。初始化、prefill、decode 与首 Token 分开计时，吞吐使用实际生成数量和对应解码耗时计算。

基线与候选的完整 Token ID 应分别保存为每行一个整数的文件，再运行仓库中的比较工具：

```bash
python3 correctness/token_compare.py --baseline /path/to/baseline_ids.txt --candidate /path/to/q_vrl_ids.txt --json
```

核对两份文件的长度均等于实际输出 Token 数量，避免把空文件或截断片段的匹配当成完整验收。该工具的 `divergence_step` 使用零起始索引；报告第几个 Token 时需明确索引口径。

## 6. 已淘汰路线与后续范围

add-y4 及 add-y4 + q-VRL 的结果文件记录第 187 步发散，不能以短序列匹配或更高吞吐替代长序列精度验收。其他负结果包括朴素 y4、全层 VRL、激进展开、静态回放及部分 LM-Head 变体，历史说明见 [失败尝试](docs/failed_attempts.md)。其中崩溃、无收益与数值发散应分别判断；电压跌落、bank conflict 或驱动内部缺陷等原因仍需与对应设备日志、profiling 或诊断证据核对。

仓库另有独立 LM-Head 微基准，覆盖 Stage A–E。小 K 批验证与推测解码属于研究方向，现有公开结果没有提供该路线的端到端加速验收。

本地网页联调文件尚未纳入当前已发布代码。19.94 的短上下文解码归档不能替代网页首字、长上下文或超过 512 Token 生成的测量；现有公开归档没有提供这些场景的持续运行与首字延迟结果。

## 7. 目录与文档

| 路径 | 内容 |
|:---|:---|
| [hook/](hook/) | DLSYM Hook 与构建入口 |
| [kernels/](kernels/) | 原始、INT4 和虚拟车道候选内核 |
| [scripts/](scripts/) | 设备对照与速度测试脚本 |
| [correctness/](correctness/) | Token 与张量比较工具 |
| [experiments/reproducible/](experiments/reproducible/) | 参考组结构化结果 |
| [experiments/tuned/](experiments/tuned/) | 异步、Gold 与失败候选结构化结果 |
| [experiments/final_validation/](experiments/final_validation/) | 长生成与候选验证汇总 |
| [benchmarks/lmhead_microbench/](benchmarks/lmhead_microbench/) | 独立 LM-Head 微基准 |

[环境对照矩阵](docs/benchmark_matrix.md)、[最终方法选型](docs/final_method_selection.md)、[单 Token 报告](docs/a540_minicpm_single_token_report.md)和 [架构说明](docs/architecture.md)保留了历史分析。部分频率、吞吐、正确性范围及带宽归因尚未统一；查阅性能时应结合本 README 的证据范围和具体实验文件。

## 开源协议

本项目采用 [Apache-2.0 License](LICENSE)。
