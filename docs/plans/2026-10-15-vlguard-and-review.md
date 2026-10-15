# VLGuard 接入、代码审查与执行手册

**Goal:** 补齐第三组独立 LoRA SFT，验证本地数据/训练契约，并单独记录数据处理和实验特殊约定。

**Architecture:** 已验证的 `data/step1/prepared` 保持原样；VLGuard 使用独立训练根 `data/step1/vlguard`。从父根的已锁定 train.json/train.zip 转换；准备时与父根的全部评测图片/问题检查精确重复。所有模型仍在父根同一评测集上评测。

**Tech Stack:** Python 标准库/Pillow；单卡 BF16 Transformers 4.57.1、PEFT 0.17.1，依赖在服务器验证。

用户已授权本地编辑、只读数据审查及离线测试。真实下载/准备、Git 同步、SCP、服务器安装和模型运行由用户执行；不创建 worktree、不自动提交、不使用子代理。

1. `src/mm_sft_lab/vlguard.py` 与 `scripts/prepare_vlguard.py`：严格解析 instr-resp，一对问答一个单轮样本；每个样本只监督响应，原图可复用。验证 safe/unsafe 数据形状、图片路径、ID；记录源 commit/哈希、父评测 manifest 哈希、全量/过滤/类型数量和输出哈希。只使用 train，不加载 test 或 LLaVA mix。
2. `configs/train/vlguard_lora.json` 与 `training_data.py`：复用 16/32/0.05、5e-5、1 epoch、有效 batch 32、4096、视觉完全冻结配方；支持单图 VLGuard，text-only HH 契约保持。
3. 新增离线测试：安全图的两问答都保留、输入无参考泄漏、路径/重复/空响应拒绝、评测污染过滤、冻结快照幂等、训练 plan 和监督 mask。支持合成端到端验收第三组。
4. `scripts/audit_step1.py`：只读检查已准备快照的文件/图像哈希、数量守恒、训练与评测重复；分析 HH 未解析分支的精确原始上下文是否与保留训练行匹配。输出独立报告，不改现有数据或静默恢复异常对话。
5. 审查训练循环的梯度累积、尾 batch、warmup、视觉 eval、checkpoint/RNG 恢复及 adapter 重载身份；必要修复配回归测试。GPU 数值等价性不以 mock 验证替代。
6. 单独文档 `docs/data-and-experiment-notes.md` 记录事实、处理规则、与原始协议的差异、预算不对齐、指标分母、剩余限制；`docs/server-runbook.md` 分阶段列检测、audit、smoke、B0、训练与 adapter 复评命令。
7. 完整离线测试与三组合成流程；仅只读审查真实快照。交付文件、验证证据与用户回来后的首个执行阶段。

上游格式与转换依据：[VLGuard 官方转换脚本](https://github.com/ys-zong/VLGuard/blob/main/convert_to_llava_format.py)。这里选择独立问答，避免将另一条安全目标响应作为训练上下文；样本数按实际问答计，不把 2,000 图像称作 2,000 optimizer 样本。
