# HH 真实数据格式审计与准备修复

目标：让已下载的官方 harmless-base 数据进入现有准备阶段；不把异常历史或空目标作为 SFT 样本，不因一个坏样本中断全量准备。

架构：保持 `parse_hh` 与默认 `hh_rows` 严格报错。准备阶段显式启用逐行排除记录，保留原始零基 source index、一基行号、原因及错误详情。test 不用于训练，仅提取 chosen/rejected 两支可解析的用户上下文；替换最终回答作解析占位，使空最终回答仍可用于去污染。无法解析的 test 分支单独记录。

依据：2026-10-15 本地只读扫描，train 原始 42,537 行，严格格式不合格 279 行、历史不一致 2 行；test 原始 2,312 行。TextVQA 标注数量和图片 ZIP 目录、WildJailbreak eval 类别数量与准备入口一致。

执行：助手本地修改与离线测试；用户执行真实数据解压、准备和 plan。沿用当前 checkout，不创建 worktree、不自动提交、不使用子代理。

1. 修改 `src/mm_sft_lab/step1_data.py`：可选异常收集；test 上下文独立提取；原始/保留/排除数量守恒；写入 `hh_test_context_issues.jsonl` 并纳入快照哈希。
2. 修改 `scripts/prepare_step1.py` 与 `docs/step1-guide.md`：输出原始样本数及 test 上下文异常数，说明过滤策略。
3. 在 `tests/test_step1.py` 验证默认严格、显式审计继续、异常行号、历史不一致、空 test 目标上下文恢复、准备快照重跑一致。
4. 只读扫描真实 HH 验证 42,537 = 合格对话 + 格式/历史排除；完整离线回归通过后给用户准备命令。不代执行真实准备或训练。

VLGuard 此次仍仅有下载入口；训练转换和配置另行接入。
