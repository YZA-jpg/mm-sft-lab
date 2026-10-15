# mm-sft-lab

**Multimodal SFT: Safety & Capability**

研究多模态监督微调数据如何影响模型安全性、有效性与通用能力。

第一阶段：在服务器单张 NVIDIA A100-SXM4-80GB 上，评测本地
`/data1/sharehub/models/Qwen3-VL-4B-Instruct` 的 SIUO 和 MSSBench baseline。

设计文档：[baseline 与 SFT 基础 pipeline](docs/plans/2026-10-13-baseline-and-sft-design.md)。

操作约定：[本地修改与服务器测试流程](docs/workflow.md)。

数据准备：[SIUO / MSSBench 下载与校验](docs/data-preparation.md)。

服务器数据源网络受限时：[本地下载与离线传输](docs/offline-data-transfer.md)。

已确认的数据版本与两端校验统计：[baseline 数据快照](docs/data-snapshot.md)。

当前状态：真实数据已在本地准备完成并传入服务器 `data/imported`，两端校验通过。框架提交 `fb33633` 已同步服务器并通过 83 项离线测试；真实 GPU 推理、12 条诊断和 48 条采样试跑已完成。48/48 生成成功，16 个配对完整，3 条 embodied 截断，重合 12 条回答完全复现。全量 baseline、真实 judge 和 SFT 尚待完成。

Step 1 新工作见 [数据与训练指南](docs/step1-guide.md)：TextVQA/HH/WildJailbreak 已完成真实本地准备；VLGuard 的独立训练快照入口及第三组 LoRA 配置已补齐，真实准备待用户执行。包含只读数据审查、离线测试、人工判分导入、TextVQA 评分，以及单卡 LoRA plan/preflight/audit/smoke/train/恢复入口。真实 GPU 训练和 adapter 重载尚待服务器验证；当前本地修改尚未同步服务器。

数据过滤数量、VLGuard 问答展开方式和训练/评测特殊约定独立记录于 [数据处理与实验说明](docs/data-and-experiment-notes.md)。用户回来后的分阶段命令见 [服务器执行手册](docs/server-runbook.md)，先执行本地 VLGuard 准备阶段。

启动说明：[Baseline 框架与服务器流程](docs/baseline.md)。

当前配置与下一步：[采样 baseline 与服务器实验记录](docs/sampling-baseline.md)。原始诊断入口：[smoke 与批量检查](docs/smoke-tests.md)。

已实现的数据流程：

```text
官方数据 → sources.lock.json 固定版本 → raw/ 原始文件
        → 图片、题目、配对校验 → prepared/inputs.jsonl
                             → prepared/evaluation.jsonl
                             → prepared/manifest.json
```

`inputs.jsonl` 保存待评测的文本与图片路径，`evaluation.jsonl` 单独保存评分标签与参考答案。数据准备阶段不修改官方题目、选项或图片，不依赖 GPU 和 PyTorch。

评测入口：

```bash
python scripts/lab.py --help
# preflight → plan → generate → validate-run → judge → score → compare
# compare-batches：同清单、仅改变 batch size 的生成结果比较
```

配置位于 `configs/`：`baseline_smoke.json`、`baseline_smoke_batch4.json`、`baseline_small.json`、`baseline_full.json`、`siuo_3sample.json`、模型 API 与 judge 示例，以及明确标记的 `mock.json`。API 密钥从环境变量读取；真实服务地址和模型名待服务器运行时配置。

已验证的单次采样设置另存为 `baseline_sampling_smoke.json`、`baseline_sampling_small.json`、`baseline_sampling_full.json`：batch 1、2048 新 token、temperature 0.7、top_p 0.8、top_k 20、基础 seed 42。MCQA 可直接本地评分；开放回答的人工、本地或 API 判分方式尚待确定。

在仓库根目录运行离线检查：

```bash
python -m unittest discover -s tests -v
```

依赖安装与服务器操作顺序见数据准备文档；Git 提交、推送与服务器命令由用户执行。
