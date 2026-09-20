# OPEN_SOURCE_PLAN.md: AdrenoLLM 开源发布与社区维护规划

## 1. 仓库定位与愿景 (Repository Mission)
**一句话定位**：  
*Architecture-aware INT4 LLM inference optimization for legacy Qualcomm mobile GPUs.*  
让数十亿存量老旧移动设备能够稳定、高吞吐、高质量地运行开源小参数大语言模型，并为学术界提供一套透明、可审计、拒绝虚假性能指标的端侧系统实验 Artifact。

---

## 2. GitHub 目录结构与规范 (Repository Layout)
```text
AdrenoLLM/
├── README.md                 # 首页概览、复现表格、核心思想与负结果声明
├── LICENSE                   # Apache 2.0 (内核) / MIT (运行时与工具)
├── FINAL_RESULTS.md          # 权威基准与审计结果
├── ARTIFACT_STATUS.md        # 论文复现性声明
├── PAPER_MATERIAL.md         # 论文实验数据与核心贡献
├── docs/
│   ├── architecture.md       # Adreno 540 架构与 VRL 原理解析
│   ├── benchmark.md          # 详细基准测试复现指南
│   └── failed_attempts.md    # 负结果清单 (Static Replay, Unroll, etc.)
├── kernels/
│   ├── int4/                 # Block-32 INT4 高速解码算子
│   └── vrl/                  # Virtual Reduction Lane 核心 OpenCL 内核
├── runtime/
│   └── hook/                 # libadrenollm_dlsym_hook 源码与 Android.mk
├── correctness/
│   ├── token_compare.py      # Token 序列全量比对脚本
│   └── tensor_compare.py     # FP16 Tensor 误差分析脚本
├── benchmark/
│   ├── run_sampling_bench.sh # 5 次稳定复测脚本
│   └── monitor_thermal.sh    # CPU/GPU 调频与温度监视脚本
├── experiments/              # 结构化实验数据库 (YAML + JSON + README)
└── paper/
    ├── reviewer_attack.md    # 审稿人防御分析
    └── figures/              # 论文图表矢量图与渲染脚本
```

---

## 3. 发布时间线 (Release Milestones)

| 阶段 (Phase) | 任务目标 (Deliverables) | 时间节点 |
| :--- | :--- | :--- |
| **Phase 1: Internal Freeze** | 实验审计完成，锁定 16.98 tok/s 稳健构型，封存负结果 | Day 0 (已完成) |
| **Phase 2: Code Cleansing** | 剥离硬编码路径，整理 OpenCL 内核注释，完善 Python 测试脚本 | Day 3 |
| **Phase 3: Public Beta** | 创建 GitHub 仓库，发布 README，上线 `experiments/` 数据库 | Day 7 |
| **Phase 4: Artifact Evaluation** | 提交论文并开启 ACM/IEEE Artifact Evaluation 独立复现评测 | 论文投稿期 |

---

## 4. 社区维护与 Issue 治理准则
1. **真实性优先原则**：任何针对性能的 PR 必须附带：
   - 至少 5 次实机测量的 Mean ± Std。
   - 64-token 与 256-token 的 Token ID 一致性比对报告。
   - 设备温度及 CPU/GPU 调频设定。
2. **禁止虚假优化**：严禁提交牺牲数值精度以换取虚高吞吐量（如产生 trajectory mismatch 的 Naive 向量化）的代码。
