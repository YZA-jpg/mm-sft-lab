# B0 主表收尾：WildJailbreak 与 TextVQA Val

> 文档状态（2026-10-23）：历史 B0 协议收尾记录。当前 v2 baseline 请使用 [当前手册](baseline-runbook.md)。

2026-10-20 用户确认：SIUO 保留三条平台缺失，只报告当前分数与覆盖率；能力主指标采用 TextVQA Val，Test 保留待官方评分。接下来仅新增 WildJailbreak harmful 2,000 条、benign 210 条语义判断，不重新推理或重判已成功的 SIUO/MSSBench。

## 主表范围和已有结果

| 任务 | 指标 | 已确认 B0 结果 | 当前状态 |
| --- | --- | ---: | --- |
| SIUO gen | 不安全率 | 89/164 = 54.2683% | 164/167 有效，三条缺失保留 |
| MSSBench chat | unsafe 危险服从率 | 437/600 = 72.8333% | 1,200/1,200 已判 |
| MSSBench embodied | unsafe 危险服从率 | 241/380 = 63.4211% | 760/760 已判 |
| WildJailbreak harmful | ASR | 待本轮判分 | 2,000 条已生成 |
| WildJailbreak benign | 过度拒答率 | 待本轮判分 | 210 条已生成 |
| SIUO MCQA | 准确率 | 84/167 = 50.2994% | 完整 |
| TextVQA Val | VQA soft accuracy | 80.9420% | 5,000 条，完整 |

SIUO 全分母缺失取值范围为 89/167–92/167（53.2934%–55.0898%），不是置信区间。MSSBench 两类场景分开，不合并为一个未预注册的平均分。截断回答保留，按原预算评分。TextVQA Val 不是不安全率；Test 已导出 5,734 条，不将 Val 分数填为 Test 分数。

## 实现与身份

- `complete_b0_server.sh` 是用户运行的启动器，支持 start / resume / status / report。
- 沿用服务器 `runs/recovery-tools/judge_partial_v3.py`，SHA256 为 `5831d296fe06b9d4bd57622f5f0d3d322a9faa904fe352e163ce84b99afe811b`。不上传新 judge 引擎、不改 src、不修改环境。
- 从完整 main 捕获新快照；同一配置/服务/rubric/脚本/运行指纹和回答哈希校验通过后，导入 2,124 成功＋2 审核拒绝＋1 文件超限。新语义总数 4,337，剩余请求 2,210。
- 两条前台检查均成功后，后台续判剩余 Wild 请求。已有标签不重复发送，已知平台拒绝不重试。新普通错误仍记录并按既有故障规则停止，后续使用 resume。
- 新目录为 `runs/B0-full-20261016T091009Z/completion-*`，新指针为 `latest-b0-completion.txt`。原 `latest-completed-judge.txt` 和其指向的 SIUO/MSSBench 目录保留。
- `report_b0.py` 独立读取 journal 重新聚合，校验原始标签、快照/来源、保存指标与签名基础，导出 `b0-table.json`、`b0-table.csv`、`b0-table.md`。不写回原生成或旧 judge 文件。
- 主表收尾要求 Wild 两个 split 均完整，MSS/MCQA/Val 完整，SIUO 完整或恰好是用户确认的三条平台缺失。其他缺失不会被默认为已接受；表中保留 complete 与 coverage。
- 合成演练必须显式使用 allow-mock；导出写明 SIMULATION ONLY，real_baseline_verified=false，不作为真实测量结果。

## 阶段一：本地上传两个操作文件

在本地 PowerShell 执行：

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
scp ".\scripts\complete_b0_server.sh" ".\scripts\report_b0.py" "tstar_001:/data1/yaoziao/mm-sft-lab/runs/recovery-tools/"
if ($LASTEXITCODE -ne 0) { throw "收尾工具上传失败" }
```

## 阶段二：服务器启动 WildJailbreak 判分

```bash
cd /data1/yaoziao/mm-sft-lab
bash runs/recovery-tools/complete_b0_server.sh start
```

先进行无 API 请求的快照与复用计划检查，预期计数为 expected=4337、reuse_successful=2124、reuse_blocked=2、reuse_input_limit=1、remaining=2210。通过后隐藏输入 Key，前台检查两条，成功后 nohup 继续。密钥只进入进程环境，不写文件、不发到聊天。这个阶段不加载 GPU 模型。

成功启动会显示 completion 目录、worker PID 和表格路径。上述是待用户执行的流程，不表示本地已经真实调用 API 或服务器已经启动。

## 阶段三：查看和恢复

```bash
cd /data1/yaoziao/mm-sft-lab
bash runs/recovery-tools/complete_b0_server.sh status
```

status 只读取最新完整 journal 行与日志，不发 API。日志和实时 Wild 数量反映进行中的进度；保存 manifest 在本轮结束时更新。后台结束会自动重新验证并导出主表，status.txt 为 COMPLETE B0 safety and TextVQA Val 时表示约定的主表收尾条件满足，三条 SIUO 缺失及 Test pending 仍会保留。

若停止，先检查错误与进程；锁仍存在时不直接删除。修复认证或临时故障后，沿用同一目录：

```bash
cd /data1/yaoziao/mm-sft-lab
bash runs/recovery-tools/complete_b0_server.sh resume
```

resume 跳过成功记录和已知平台限制，不重新创建快照，不重复 start。若出现新的 Wild 平台缺失，工具保留并输出 coverage，主表收尾条件不会自动视作满足，需依据真实返回另行说明。

## 阶段四：取得最终主表

```bash
cd /data1/yaoziao/mm-sft-lab
bash runs/recovery-tools/complete_b0_server.sh report
```

report 只读取/校验本地文件并导出表格，无需 Key，不调用 API。输出的 Markdown 表可直接回传聊天；同一任务的 table/ 文件夹中保留 CSV/JSON，记录分子、有效分母、完整分母、截断、平台缺失和源文件 SHA256。

最终主表只有 B0 实测行；尚未训练的 TextVQA/HH/VLGuard adapter 不填虚构数值。Test 状态保留待官方评分，不作为本轮主表收尾的阻断条件。

## 离线验证记录

2026-10-20：新增四项合成演练与全套 193 项离线测试通过。验证同 fingerprint 的早期快照到完整快照复用、仅新增 Wild 调用、已知平台拒绝保留、旧 judge 目录逐文件字节不变、重复 resume 零新增请求、保存指标篡改拒绝、运行锁拒绝、adapter 不得标为 B0、mock 标记和 Test pending 保留。Bash 语法及嵌入 Python 语法检查通过，v3 引擎 SHA256 保持不变。

这些是本地离线验证，不是服务器上的 Bash worker/API 执行结果。真实 Wild 分数与最终表格以用户执行回传为准。
