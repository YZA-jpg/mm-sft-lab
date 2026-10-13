# mm-sft-lab

**Multimodal SFT: Safety & Capability**

研究多模态监督微调数据如何影响模型安全性、有效性与通用能力。

第一阶段：在服务器单张 NVIDIA A100-SXM4-80GB 上，评测本地
`/data1/sharehub/models/Qwen3-VL-4B-Instruct` 的 SIUO 和 MSSBench baseline。

设计文档：[baseline 与 SFT 基础 pipeline](docs/plans/2026-10-13-baseline-and-sft-design.md)。

操作约定：[本地修改与服务器测试流程](docs/workflow.md)。

当前状态：已整理设计文档、操作流程和 Git 基础配置；数据下载、推理、判分和训练入口尚未实现，服务器实验尚未运行。
