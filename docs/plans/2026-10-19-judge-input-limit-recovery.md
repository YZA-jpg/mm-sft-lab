# SIUO Judge 输入文件超限恢复计划

**Goal:** 保留已完成的 MSSBench 与 1,990 条成功判分，将 SIUO 图片超限独立记为 judge 缺失，继续剩余 134 条。

**Architecture:** 只更新 scripts 操作工具，不修改 src、原图或模型回答。新版本用新签名/新目录显式导入已知 v2 签名的结果，保存迁移来源；保留 v1 迁移兼容。输入文件限制和内容审核分别分类、分别统计。

**Tech Stack:** Python urllib/unittest、既有 metrics、用户执行的 Bash 启动器。

## 事实与范围

用户最新回传：MSSBench chat 1,200/1,200、embodied 760/760 均完整；危险服从率分别 0.7283333333333334、0.6342105263157894。SIUO gen 30 条成功、2 条审核拒绝、1 条文件超限、134 条未尝试。超限样本 siuo/gen/2014，HTTP400，code=invalid_parameter_error，type=invalid_request_error，message 为 Multimodal file size is too large。TextVQA Val=0.8094200000000044，MCQA=0.5029940119760479，Test 5,734 条已导出、官方分数待定。

## 实施与验证

1. 精确识别上述状态/码/类型/消息组合为 provider_input_limit，labels=null；一般 invalid_parameter_error 仍停止。两个平台缺失类别均不触发服务故障连续三错，不自动重试已知拒绝或超限。
2. 固定旧 v2 字节 SHA256=e2525f3a727d6466ace7f6a4820891eaee20d8b49b1bbdd6be16a3e171ff9ca3，验证旧 signature_basis、服务/配置/rubric/来源/回答与原始标签后迁移。预期复用 1,990 成功＋2 审核拒绝＋1 文件超限，仅剩 134 个请求需尝试，MSSBench 不重判。
3. 增加输入超限计数、单独清单与联合平台缺失清单，processing_complete 按成功＋审核拒绝＋文件超限计算，scoring_complete 仍只接受全部有效评分；只要有文件超限，处理结束状态为 finished_with_provider_errors。
4. 新增用户执行恢复启动器，从 v2 manifest 读取原 snapshot/probe 路径，先 plan-only 再 nohup；不重建原运行、不压缩/缩放图片、不改 API 输入或直接改旧签名。
5. 更新独立错误说明及数据处理文档，新增当前可汇报指标摘要；明确 SIUO 17/30 只是当前子集、平台缺失不算模型不安全、不同 SFT 版本需统一覆盖口径。

测试需覆盖连续超限继续、无关400停止、v2成功/审核/超限迁移零重复调用、篡改拒绝、分母和状态语义、v1兼容；运行全套离线回归与 Bash/diff 检查。真实恢复由用户执行，助手不 SSH 或调用 API。

验证结果：23 项操作工具测试通过，全套 189 项离线测试通过，恢复启动器 Bash 语法、CLI 参数与 git diff --check 通过，src 无修改。额外加载已保存、哈希匹配的原 v2 脚本实际执行模拟文件超限，再由 v3 迁移，确认成功标签复用、输入超限分类、旧文件字节不变且迁移阶段零重复请求。所有验证均无真实 API 调用。

交付为 scripts/judge_partial.py 上传至服务器 judge_partial_v3.py，以及 scripts/recover_input_limit_judge_server.sh。原 v2 文件和目录保留；新目录 judge-input-recovery-*，当前只确认工具就绪，尚未收到真实恢复回传。
