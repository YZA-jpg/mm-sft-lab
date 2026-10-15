# 后续 SFT 配置

此处登记数据配方、训练预算、LoRA 参数和训练环境。`textvqa_lora.json` / `hh_lora.json` / `vlguard_lora.json` 提供同一显式起始配方，训练入口为 `scripts/train.py`；真实 GPU 训练尚待服务器验证。

每个配方需要记录训练数据来源/哈希、评测集去污染结果、assistant 监督 token 数、视觉模块冻结策略、训练种子、起始模型和输出 checkpoint；评测复用现有 generation/judge/score 入口。

2026-10-15 已补齐三组独立训练入口：分别从同一 base 初始化，各自完整可用训练划分 1 epoch，作为探索实验。TextVQA/HH 使用父数据根，VLGuard 使用 `<父根>/vlguard` 独立训练快照；评测都使用父根。处理规则、全量与可用数量、上游差异及限制见 [独立实验说明](../../docs/data-and-experiment-notes.md)。

用户参数保持 LoRA 16/32/0.05、LR 5e-5、AdamW、有效 batch 32、总序列 4096。新增 scheduler/warmup/weight decay、target modules 和图像预算是本轮显式起始值，服务器测试前核对。分阶段执行与命令见 [Step 1 指南](../../docs/step1-guide.md)，先 plan/preflight/audit，再分别运行 smoke 和 train。
