# Judge 内容审核拒绝的恢复计划

**Goal:** 单个样本被 API 内容审核拒绝时记录缺失、继续其他样本，先完成 MSSBench，同时保留已有 7 条真实成功判分。

**Architecture:** 仅更新 scripts 操作入口和文档，不修改 src 推理包/训练环境。旧目录只读保留，使用新执行签名、新 judge 目录及显式旧版迁移；验证来源运行、回答哈希、rubric/配置/服务和已知旧脚本签名后导入成功结果。已知内容审核错误导入为 blocked、labels=null，不再次请求同一已知被拒样本。

**Tech Stack:** Python 标准库 urllib/unittest、原有聚合函数、Bash 用户执行启动器。

## 实施步骤

1. 在 `scripts/judge_partial.py` 结构化保存 HTTP 状态与平台错误码。仅 HTTP 400 且结构化 code/type 均为 data_inspection_failed 识别为 provider_content_filter。其他 400、401/403/404 继续立即停止，临时错误连续三次停止。
2. 将内容审核拒绝保持为 error 记录，增加 error_kind/provider_error_code；不产生安全标签，不计为成功，不触发服务故障连续计数。默认续判不重试同一 blocked；按模块顺序先 MSSBench 再 SIUO。
3. 显式 `--reuse-legacy-judge-dir` 校验已知旧脚本 SHA256 和旧签名，再验证每行 SHA/身份/标签。成功与已知 blocked 导入新目录，逐行保存原签名、原行摘要和来源文件摘要。其他错误仍待显式重试。禁止修改旧 manifest 伪装签名。
4. 为 manifest 和模块报告增加处理完成/评分完成区分、blocked 清单和覆盖率；only-blocked 时 processing_complete=true、scoring_complete=false、status=finished_with_blocks，不误标完整 baseline。保留原聚合的全分母与 valid-only 率。
5. 新增用户执行恢复启动器，使用原冻结快照/原视觉探针/新目录，先离线 plan 验证并确认复用数量，再 nohup 继续。助手不 SSH、不调用 API、不改动原推理任务。
6. 单独说明该案例并链接进 `docs/data-and-experiment-notes.md`，给学长汇报时区分推理输出、judge 平台审核和模型安全判断，说明缺失非随机的潜在偏差与跨模型统一覆盖要求。

验收：结构化拒绝/普通400/认证错误/连续拒绝/临时错误均覆盖；旧版迁移复用7成功+1blocked后零重复请求；服务、回答、rubric或旧签名变化拒绝复用；blocked不计入unsafe/safe标签。运行新增离线测试、全套回归、Bash语法和diff检查，真实恢复待用户执行。

验收结果：19 项操作工具测试通过，全套 185 项离线测试通过；两份 Bash 启动器语法与 diff 检查通过。额外加载字节哈希为 `204e962fd08d367954ac450f6f89b4702e8c4193d8523e14b0150af5965148b8` 的原 v1 脚本实际执行模拟中断，再由 v2 校验/迁移/完成其他模拟请求，兼容性演练通过，旧目录字节保持一致。所有验证均无真实 API 调用。

服务器交付：新 helper 上传为 runs/recovery-tools/judge_partial_v2.py，保留原 judge_partial.py；恢复启动器使用原 snapshot 与视觉报告，在新的 judge-recovery-* 目录处理。plan-only 预期显示 7 条成功复用、1 条审核拒绝复用、2,119 条仍需尝试。尚未收到真实恢复启动或完成回传。
