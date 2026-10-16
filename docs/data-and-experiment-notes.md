# 数据处理与实验特殊约定

更新：2026-10-15。此文档是本项目的数据处理和实验解释记录，独立于操作手册。实施入口见 [Step 1 指南](step1-guide.md)，后续阶段见 [服务器执行手册](server-runbook.md)。

## 1. 研究目标与四个独立模型

探索 Qwen3-VL-4B-Instruct 在 TextVQA 良性 OCR 任务 SFT 后，安全 benchmark 的不安全率是否变化；VLGuard 与 HH harmless chosen 提供多模态/文本安全数据对照。结论限定于当前模型、数据和训练配方，不预设一定出现安全下降。

| ID | 起点 | 训练数据 | 模态 |
| --- | --- | --- | --- |
| B0 | 原始 checkpoint | 无 | 原始模型 |
| T1 | 同一原始 checkpoint | TextVQA train | 图像＋文本 |
| V1 | 同一原始 checkpoint | VLGuard train，全部三类问答 | 图像＋文本 |
| H1 | 同一原始 checkpoint | HH-RLHF harmless-base train chosen | 纯文本 |

三个 adapter 独立初始化，不串行续训、不混合三个训练集。用户选择“每组完整可用训练划分 1 epoch”，属于预算未对齐的探索实验；数据清洗后完整训练不等于原始行数全部入训。

## 2. 来源、冻结与当前验证状态

原始数据和图像不进入 Git。HF 每源独立固定 commit，记录文件 SHA256；TextVQA 使用固定官方 v0.5.1 URL，以内容哈希冻结，不能将 URL 版本称为 Git commit。下载失败不更新已锁定版本，也不阻止独立来源继续下载。单源重试成功后，历史批量汇总可能仍显示旧失败；以逐源 manifest 和重新校验为准。

| 来源 | 当前数据范围 | 状态 |
| --- | --- | --- |
| SIUO/MSSBench | 原有锁定快照 | 已在本地和服务器校验 |
| TextVQA | train/val/test 标注与图片 | 用户已完成真实准备 |
| HH | harmless-base train/test；只训练 train chosen | 用户已完成真实准备 |
| WildJailbreak | 官方 eval，未取 train | 用户已完成真实准备 |
| VLGuard | train.json/train.zip/README；不取 test、LLaVA mix | 已下载、原始图像/标注只读检查通过；新转换代码已离线测试，真实准备由用户执行 |

各来源当前锁定版本：

| 来源 | 当前原始版本 |
| --- | --- |
| TextVQA | 官方静态 URL v0.5.1，逐文件 SHA256 见 raw source lock |
| HH harmless | `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa` |
| WildJailbreak eval | `5ddc12a7894f842b0619b8e1c7ee496b198af009` |
| VLGuard train | `b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46` |

父评测与 TextVQA/HH 训练快照：

```text
3b847683c4ca245d336108853bb49fe6d5332c84c098485d001fb7f851fe20fd
```

本地审查报告保存在忽略的 `runs/step1-readonly-audit-20261015/`。它是数据完整性报告，不是模型能力、安全分数或真实训练验收。

## 3. TextVQA 处理

每题一个训练样本，图像＋固定短答提示＋问题 → 一个确定性答案；不将十位标注者的答案展开为十个训练样本。仅监督回答和终止 token。原始十个答案继续保留在锁定的 raw JSON 中。

先排除原始空白标注；按官方评分规范化结果统计多数组，平票按规范化字符串字典序，再在该组选择频次最高的原始表述，最后按字典序处理平票。返回去除首尾空白的原始表述。

**特别处理：** `a`、`the`、`a+` 是有效 OCR 目标，评分规范化可能将它们变为空字符串。不能以规范化是否为空来删除训练答案，也不能直接把规范化字符串用作训练目标。此问题修复后，全部 34,602 条 train 标注通过目标生成检查；其中 26 条的所有标注规范化后为空，实际原始文字有效。混合标注中有效的此类多数答案也正常参与多数表决。

过滤 TextVQA train 与 val/test 同源 image ID、所有评测图片文件 SHA256，以及同图同问题的训练重复。图片 SHA256 比较的是文件字节，不是解码后像素；重编码图片可能无法检出。

| 项目 | 数量 |
| --- | ---: |
| 原始 train 问题 | 34,602 |
| 同图同问题重复排除 | 17 |
| 最终训练样本 | 34,585 |
| 最终训练独立图片（文件哈希） | 21,953 |
| 有效 batch 32 的计划更新步数 | 1,081 |

当前数据的 heldout ID/评测图片哈希过滤未额外排除样本，不表示没有执行这些检查。

## 4. HH harmless chosen 处理

仅使用 `Anthropic/hh-rlhf/harmless-base/train.jsonl.gz` 的 chosen 分支。严格按 Human/Assistant 标记解析，完整历史作上下文，只监督最后一个 chosen assistant 回答；rejected 不作为训练目标，但用于检查两支公共历史是否一致。

这是偏好数据用于 SFT 的适配。chosen 只表示相对 preferred，不保证每个回答绝对安全；不将 H1 命名成经过人工认证的安全数据集。[Anthropic 官方说明](https://huggingface.co/datasets/Anthropic/hh-rlhf)

非交替角色、空轮次、模型控制 token 或两支历史不一致的行被排除并逐条记录。记录原始零基 source index、一基行号、原因和错误详情；不会通过合并错误角色来伪造有效历史。默认底层解析器仍严格报错，只有准备入口显式开启审计过滤。

| 项目 | 数量 |
| --- | ---: |
| 原始 train 行 | 42,537 |
| 严格格式不合格 | 279 |
| chosen/rejected 历史不一致 | 2 |
| 与 test 上下文重复 | 141 |
| 最后用户问题与评测 query 精确重复 | 7 |
| chosen 对话内部重复 | 2 |
| 最终训练样本 | 42,106 |
| 有效 batch 32 的计划更新步数 | 1,316 |

test 2,312 行只用于去污染。提取两支可解析的上下文时不要求最终回答非空；最终回答被替换成解析占位，随后丢弃，不进入任何训练样本。共得到 2,306 个唯一可解析上下文。

**13 个 test 分支问题：** 来自 9 条原始记录，并非 13 条训练样本。2026-10-15 只读补充审查将这些分支的原始最终 Assistant 前缀与保留训练行比较，精确重合为 0。该检查没有修补或改写历史，不覆盖语义改写；不能宣称所有 test 上下文都已完整解析或完成语义去污染。

## 5. VLGuard 处理与上游协议的差异

原始 train 有 2,000 条图像记录，其中 safe=false 1,023 条、safe=true 977 条，合计 2,977 组问答。一个上游 ID `Ads_303871` 被两条不同记录复用，两条记录引用同一图片、问答不同。因此实际引用 1,999 个图片文件；不能把 2,000 条记录称为 2,000 个独立图片。

训练样本 ID 包含原始行号＋原始 ID＋instruction 字段名。保留原始 ID 和行号供溯源，不因 ID 复用删除有效问答。

| 问答类型 | 原始/只读候选数量 |
| --- | ---: |
| 不安全图像的 instruction → response | 1,023 |
| 安全图像的 safe_instruction → response | 977 |
| 安全图像的 unsafe_instruction → response | 977 |
| 合计 | 2,977 |

**本项目适配：** 一组问答一个独立单轮样本，共用原图，只监督该回答。上游转换脚本会随机调整安全图像的两个问答顺序并拼接为多轮对话；此处不引入随机拼接或另一条目标回答的上下文，保持与其他训练组的最后回答监督方式一致。这是项目协议，不冒充上游完整复现。[官方转换脚本](https://github.com/ys-zong/VLGuard/blob/main/convert_to_llava_format.py)

严格检查字段类型、问答种类、非空响应、路径和图像可读性；过滤评测图片 SHA256、评测 query 精确重复，以及训练同图同问题重复。只读 ZIP 检查已验证所有引用的 1,999 个图像文件，候选 2,977 组问答未触发这些过滤。正式数量以用户运行准备命令后的 manifest 为准；若仍为 2,977 条，计划 94 个 updates，最后一组只有 1 个样本。

VLGuard 使用独立训练根 `data/step1/vlguard`，父根的已验证 prepared 不改动。子快照记录父评测 manifest 的 SHA256；所有四个模型仍在父根同一套评测数据上比较。VLGuard test 未下载，不声称检查过其 train/test 重复；也不将它加入用户指定的主 benchmark 表。

## 6. 共同去污染的边界

已实施的检查是确定性的文件字节 SHA256、同源 image ID、空白/大小写规范化后的文本精确重复。训练与评测的输入/标签分开保存，训练入口不读取评测标签来构造目标。

尚未实施：图片感知近似重复、语义改写重复、基座预训练污染审计，以及人工逐条确认安全目标质量。不同数据源规模、质量和模态存在混杂因素。数据过滤日志保留在 prepared；改变规则必须建立新快照，不能覆盖已有训练用快照。

## 7. 训练配方与预算解释

用户指定：LoRA rank 16 / alpha 32 / dropout 0.05、LR 5e-5、AdamW、1 epoch、有效全局 batch 32、总序列 4096；整个视觉 encoder、projector/merger 及视觉连接层冻结。

代码的显式起始值：语言 attention q/k/v/o LoRA、bias=none，micro-batch 1 × 累积 32，BF16/SDPA，seed 42；linear scheduler，warmup=floor(steps×0.03)，weight decay 0，betas 0.9/0.999，epsilon 1e-8，gradient clip 1。这些补充值尚待首次服务器训练前核对，不能表述成已实测最优参数。

训练图像像素面积预算 65,536–1,048,576。4096 包含图像 token、模板、历史、问题和回答。全量 processor audit 超长即报错并给出清单，不静默截断、丢弃或改变分辨率。实际长度和 GPU 可运行性尚待服务器验证。

loss 只覆盖最后回答及终止 token；用户、历史、图像 token 均 mask。每个 micro loss 按实际监督 token 数加权，累积后统一除以该 window 的监督 token 总数；不把长度不同样本的均值简单平均。尾部不足 32 的 window 保留并按实际 token 归一化。

三组每 epoch 的 updates、warmup、输入/监督/视觉 token 数均不同。至少汇报可用样本、图像、updates、实际监督 token、耗时和显存；后续需预算匹配与多 seed 才能增强归因。当前 seed 42 单次探索不能代表稳定总体效应。

代码检查可训练参数只在语言 LoRA；视觉同时保持 eval；训练前后对冻结参数哈希，检查 adapter 更新与保存 tensor 一致性。checkpoint 保存 optimizer/scheduler、样本偏移、Torch/CUDA/Python RNG。token audit 绑定配置、数据、模型、代码与依赖版本，CPU audit 的 GPU 可见性可与训练不同。GPU 恢复数值等价和真实 adapter 重载尚未验证。

## 8. 评测与指标特殊约定

所有模型使用相同评测输入、模板、生成参数、judge 协议和冻结代码/依赖。当前生成协议 batch 1、max_new_tokens 2048、sampling temperature 0.7/top_p 0.8/top_k 20/seed 42。既往 batch 1/4 输出存在变化，正式对照固定 batch 1。2048 是回答长度上限，与训练总序列 4096 不同。

`step1_full.json` 9,504 请求：旧 SIUO/MSSBench 2,294＋WildJailbreak 2,210＋TextVQA Val 5,000。Test 单独 5,734 请求；全部准备数据为 15,238 请求。历史 48 条只用于运行检查，不填入全量 baseline 表。

| 任务 | 主指标 | 特殊说明 |
| --- | --- | --- |
| SIUO gen | 不安全率 | 保留 effectiveness、safe-and-effective 与判分覆盖率 |
| SIUO MCQA | accuracy | 本地选项解析，不需模型 judge |
| MSSBench chat/embodied | 不安全场景 class 2 危险服从率（项目派生） | class 3 为非安全原因能力/信息不足；不得把 1−官方 accuracy 当成纯不安全率；另报官方指标 |
| WildJailbreak harmful | ASR | 2,000 条，与 210 条 benign 的过度拒答率分别报告；项目 adapted judge 待人工校准 |
| TextVQA Val | 官方多参考 VQA accuracy | 能力指标，不是不安全率；固定短答提示，不外挂 OCR 文本 |
| TextVQA Test | 官方提交评分 | 隐藏答案，当前仅导出预测，不伪造本地准确率 |

SIUO gen 的自动 judge 需要视觉输入；MSS/Wild 当前 rubric 使用文本。接口可由本地模型服务提供，也可人工标签导入，并非必须外部付费 API。judge 服务、模型版本与边界样本校准尚未冻结。当前 MSS rubric 的上下文使用和 Wild adapted rubric 的有效性仍需人工核对，标签不能由关键词命中冒充语义判断。

截断、空回答、错误、未判分均单列。保留截断回答，不能自动加 token、选择更有利重试或当成安全回答；如修改生成协议，四个模型共同重跑。主分数报告 coverage 和分母；不将缺失/unknown 计作安全。

原始模型权重身份目前使用配置内容哈希＋权重文件 size/mtime 清单，不是所有权重字节的 SHA256；adapter 权重使用文件 SHA256。更强的基座内容证明需要另行完整哈希。旧 48 条环境与新冻结版本不直接混表。

当前 `compare` 输出逐样本改善/退化 ID 和指标差值，属于描述性配对比较，尚无置信区间或按原始场景重采样。率指标以 0–1 存储，差值乘 100 后才是百分点变化；不得把描述性差异称作显著性检验。

## 9. 本地验收记录

2026-10-15：152 项离线测试通过；三组合成数据完成准备、只读审查、训练 plan、mock 推理/判分及报表。输出 `runs/step1-three-group-final-check-v2/summary.json` 明确标记 `is_mock=true`、`real_training_verified=false`。

真实父数据只读审查为 `passed_with_notes`、`violations=[]`；VLGuard 的原始 ZIP/标注检查通过，但没有代用户生成真实 VLGuard prepared。训练代码审查另加强 token audit 的代码与依赖绑定，并以离线回归验证环境漂移会被拒绝。VLGuard 重准备时会在解压前拒绝不同的源/评测快照，避免先覆盖冻结图片再报错。梯度累积、冻结、尾 batch、调度、恢复和 adapter 身份已作静态/契约检查，GPU 数值验收仍待服务器执行。

当前没有完整 B0、安全判分或真实 SFT 结果；以上是已实现处理、用户选择和待验证协议的记录。

2026-10-16 judge 服务补充：用户提供 DashScope 的 `deepseek-v4.1-flash` 配置；已登记 [独立连接说明](judge-dashscope.md)，服务与图片输入尚未实际验证。先传输数据，优先在本地调用 judge；不因此修改已冻结的生成参数或数据快照。

## 10. 服务器导入与训练环境准备记录

2026-10-16 用户已推送并在服务器同步 `b621c47`，上传 `step1-all-20261016-004857.tar.gz`（约 16 GB）并通过 SHA256；导入根 `data/step1-imported-v1` 的三组只读审查通过，`violations=[]`。VLGuard 子 manifest 为 `12861f4c9e968ba45b1ca30e9a56320570d68ebdbdcf2f7c5edadd9a0c2dfd71`；其引用文件路径 1,999 个，按文件 SHA256 去重为 1,996 个，样本仍为 2,977 条。

原推理 Conda 环境 `.venv` 检测通过：Python 3.11.17、torch 2.8.0+cu128、torchvision 0.23.0+cu128、Transformers 4.57.1、Accelerate 1.15.0、safetensors 0.8.0、huggingface-hub 0.36.2、Pillow 12.3.0；CUDA/BF16、Qwen 导入和 pip check 正常。随后用户已用 `--clone --offline --copy` 创建 `.venv-train`，上传并仅安装 PEFT 0.17.1 的通用 wheel；pip check 与训练 preflight 均通过，状态为 `environment_ready_not_training_verified`。此检查没有加载训练权重或更新参数。

保留检测到的 Accelerate 1.15.0 / safetensors 0.8.0，替代早期尚未实测的 1.10.1 / 0.6.2 提议；`requirements-train.txt` 已相应更新。这仅调整依赖起点，用户指定的 LoRA、学习率、epoch、有效 batch 和序列长度不变。PEFT 导入已通过，GPU 训练仍须逐阶段验证。[PEFT 0.17.1 依赖声明](https://github.com/huggingface/peft/blob/v0.17.1/setup.py)

首次 TextVQA 64 条 CPU token 预审在 processor 配置检查处停止，报 `Processor did not apply the explicit training pixel budget`，未进入样本编码或训练。定位到加载参数的兼容性缺口：Transformers 4.57.1 可在构造后单独更新 `min_pixels/max_pixels`，`size` 保持 checkpoint 值；原检查只比较后者。训练入口现统一这三处设置并要求启用 resize，像素面积 65,536–1,048,576 和总序列 4096 保持不变，评测入口不修改。[加载实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/image_processing_base.py)

新增 `train.py processor-check` 使用两张合成图片验证实际 `image_grid_thw` 对应的缩放面积，覆盖放大和缩小，且不加载模型权重、数据集或 API。字段不同步的离线复现与边界测试通过，全套 159 项离线测试通过；真实 Qwen processor 探针及三组 64 条预审须在用户同步修复后重跑。此前失败日志保留，不能记作 token 审计通过。
