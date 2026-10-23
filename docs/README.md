# 文档索引

更新：2026-10-23。当前任务为修复验证 → 新 baseline；先不向学长发确认材料。操作集中在一份手册，历史资料按用途列在下方。保留原文件路径和证据，不批量删除历史记录。

## 当前需要阅读

| 文档 | 用途 |
| --- | --- |
| [baseline 执行手册](baseline-runbook.md) | 唯一当前操作入口：预检、15 条 smoke、后台监控、验收后全量、计时导出 |
| [工作流](workflow.md) | 本地 / 服务器边界与分阶段反馈约定 |
| [数据处理与实验说明](data-and-experiment-notes.md) | 数据过滤、协议变更、异常和解释依据；按日期累计 |
| [八 benchmark 来源与适配](evaluation-v2-sources.md) | 上游来源、指标和官方实现差异 |

## 数据与未来训练资料

- [SIUO / MSSBench 数据准备](data-preparation.md)、[数据快照](data-snapshot.md)、[离线传输](offline-data-transfer.md)。
- [Step 1 数据与训练指南](step1-guide.md)、[Step 1 服务器操作记录](server-runbook.md)。训练暂缓，这些文件不作为当前 baseline 的启动指令。
- [DashScope judge 说明](judge-dashscope.md)。API 验证、视觉输入与历史缺失处理在此保留；新 smoke 仅验证生成。

## 历史诊断与汇报材料

| 文档 | 状态 / 查阅原因 |
| --- | --- |
| [旧 HF baseline](baseline.md)、[旧 smoke](smoke-tests.md)、[旧采样对照](sampling-baseline.md) | 1024/2048 token、batch 对照；解释历史行为时查阅 |
| [旧 B0 进度](b0-progress-2026-10-19.md)、[旧 B0 收尾](b0-completion.md) | 旧协议数据，不能作为新版 ASR@3 的最终结果 |
| [judge 平台拦截事件](judge-content-filter-incident.md) | 保留平台拦截、输入限制及缺失分母依据 |
| [旧混合调度全量](siuo-mss-64k-full-run.md) | 本次 500 成功后异常退出的历史操作；不再按此启动 |
| [WebP 异常与旧版兼容恢复](webp-preparation-incident.md) | 根因证据与历史恢复设计；当前改走新 smoke / 新全量 |
| [给学长的确认材料草稿](mentor-experiment-review-2026-10-23.md) | 暂存草稿，不作为本轮启动前置条件；未发送 |

## 实施计划索引

日常实施计划集中在 [plans 导航](plans/README.md)，用于追溯修改；不逐篇充当现行操作文档。当前实施计划为 [修复验证与逐数据集执行](plans/2026-10-23-fresh-baseline-validation.md)。历史数量和“当时已完成”只描述各自日期，不能代替最新运行结果。
