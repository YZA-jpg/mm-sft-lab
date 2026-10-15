# Step 1 数据与 LoRA 实现计划

目标：在既有基线框架上准备 TextVQA、VLGuard、HH harmless chosen、WildJailbreak eval；提供无需 GPU/API 的测试，以及之后在服务器执行的单卡 LoRA 代码。2026-10-15 更新：VLGuard 的独立快照转换与第三组配置已实现；真实准备待用户运行。最新数据数量和特殊约定见 [独立记录](../data-and-experiment-notes.md)。

架构：新数据使用独立 `data/step1` 根目录，不覆盖已有 `data/imported`。原始快照按数据源锁定；准备阶段合并旧评测集并保留模型输入/标签隔离。训练使用独立入口与严格配置，重用结果指纹及路径校验工具。

技术：Python 标准库/Pillow/huggingface-hub；训练依赖延迟导入 torch、Transformers 4.57.1、PEFT。直接单卡循环显式实现按监督 token 归一化的梯度累积、AdamW、保存恢复；不引入分布式、量化或 DeepSpeed。

执行遵循用户约定：助手本地编辑和离线单元测试，用户执行下载、Git 同步及服务器阶段。直接在当前 checkout 工作，不创建 worktree、不自动提交、不使用子代理。

1. 数据入口：逐源下载、HF commit/静态 URL 内容哈希、断点续传、ZIP 路径校验；顺序下载四个来源并隔离失败，VLGuard 仅选择 train 标注/图片与 README，WildJailbreak 仅选择 eval。
2. 数据转换：TextVQA 在原始非空白标注中确定性选规范化最高频组，再保留原始目标表述；评分规范化为空的 `a`/`the`/`a+` 仍保留。HH 解析 chosen/rejected 公共上下文，仅监督最后一轮 chosen；Wild eval 分类。精确文本/图片去污染与过滤日志。
3. 评测适配：新增 text-only 请求、TextVQA 多参考评分/Test 导出、Wild 语义标签契约；人工标签导入，保持 pending 与 mock 区别。
4. 训练纯逻辑：配置验证、数据及图片哈希、全局 batch 32、最后回答 mask、完整序列 4096 上限、语言 LoRA 范围。
5. GPU 入口：plan / preflight / audit / smoke / train；显式训练预算、冻结参数校验、adapter 与 optimizer/RNG 状态保存恢复；base+adapter 评测。
6. 离线验收：合成小图片/HH/TSV 的端到端 mock；错误路径、去污染、mask 边界、评分、恢复身份检查。运行完整现有回归测试。
7. 文档：按阶段给用户本地下载/准备/传输、服务器环境检测/训练 smoke/正式训练的命令。完整数据、真实模型训练和判分尚未执行时明确标注。

用户已指定：各组完整可用训练划分 1 epoch、LoRA 16/32/dropout 0.05、LR 5e-5、AdamW、有效 batch 32、总序列 4096。

本次补齐的可修改起始值：语言 q/k/v/o projection、bias none、seed 42、linear scheduler、warmup 0.03、weight decay 0、betas 0.9/0.999、epsilon 1e-8、梯度裁剪 1、micro-batch 1、累积 32、BF16/SDPA；训练图片像素范围 65,536–1,048,576。超长样本默认报错并出清单，禁止静默截断/缩图。以上为显式起始配置，未经过真实 GPU 训练验证。

主要依据：

- [TextVQA 官方下载地址与目录](https://huggingface.co/datasets/facebook/textvqa/blob/main/textvqa.py)
- [HH 官方 harmless-base 文件](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/harmless-base)
- [WildJailbreak eval 文件及访问条件](https://huggingface.co/datasets/allenai/wildjailbreak/tree/main/eval)
- [MMF TextVQA 评分逻辑](https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py)
- [Qwen3-VL processor 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
