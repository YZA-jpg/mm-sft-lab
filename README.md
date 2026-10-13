# mm-sft-lab

**Multimodal SFT: Safety & Capability**

研究多模态监督微调数据如何影响模型安全性、有效性与通用能力。

第一阶段：在服务器单张 NVIDIA A100-SXM4-80GB 上，评测本地
`/data1/sharehub/models/Qwen3-VL-4B-Instruct` 的 SIUO 和 MSSBench baseline。

设计文档：[baseline 与 SFT 基础 pipeline](docs/plans/2026-10-13-baseline-and-sft-design.md)。

操作约定：[本地修改与服务器测试流程](docs/workflow.md)。

数据准备：[SIUO / MSSBench 下载与校验](docs/data-preparation.md)。

当前状态：已实现官方数据下载、版本锁定、规范化与图片完整性校验，并提供离线测试。真实数据尚未在服务器下载验证；推理、判分和训练入口尚未实现。

已实现的数据流程：

```text
官方数据 → sources.lock.json 固定版本 → raw/ 原始文件
        → 图片、题目、配对校验 → prepared/inputs.jsonl
                             → prepared/evaluation.jsonl
                             → prepared/manifest.json
```

`inputs.jsonl` 保存待评测的文本与图片路径，`evaluation.jsonl` 单独保存评分标签与参考答案。数据准备阶段不修改官方题目、选项或图片，不依赖 GPU 和 PyTorch。

在仓库根目录运行离线检查：

```bash
python -m unittest discover -s tests -v
```

依赖安装与服务器操作顺序见数据准备文档；Git 提交、推送与服务器命令由用户执行。
