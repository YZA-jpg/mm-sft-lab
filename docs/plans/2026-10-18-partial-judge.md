# 运行中 B0 的快照判分 Implementation Plan

**Goal:** 让用户在主生成继续运行时评分完全完成的 MSSBench 和 SIUO 模块。

**Architecture:** 新增独立操作脚本，先只读捕获 journal 的完整行，校验原运行指纹、回答哈希与不可变输入快照，写入新的冻结目录。标准库 API judge 只处理所选完整模块，沿用 lab_rubric_v1 与原指标代码；不修改 src、生成目录、生成锁或推理环境。用户明确本轮不判 WildJailbreak。

**Tech Stack:** Python 标准库 urllib、unittest、现有 mm_sft_lab 校验/判分函数。

用户继续在当前仓库协作、自行执行上传和服务器命令；不创建 worktree、不自动提交、不启动远程任务。

## Task 1: 冻结运行中结果

Create `scripts/judge_partial.py`; test `tests/test_partial_judge.py`。
固定读取源 journal 开始时的字节上限，只采纳以换行结尾的完整 JSON 行；不修复源文件。保留全量 selection/evaluation/配置与原 fingerprint，另存 snapshot 元数据及文件哈希。拒绝未知 ID、重复成功、错误回答哈希和混合运行指纹。输出独立目录，不向源运行写任何文件。

## Task 2: 断点与复用

同一脚本新增 judge 子命令，所选 benchmark 必须全部成功生成；只判所选模块，使用同一 rubric 和严格 JSON 标签解析。SIUO gen 需要当前服务/模型/配置一致的通过探针报告，并核对实际评测 manifest 与图像哈希。使用独立 judge 锁与 append-only journal，支持两条连通性检查、续跑和显式 retry-errors。不同快照可显式复用同 source fingerprint、judge signature、回答 SHA256 的成功判分；其他组合拒绝，避免重复收费或误复用。错误次数/超时/服务身份/脚本 SHA256 明确记录，400/401/403/404 或连续三错停止。

## Task 3: 覆盖率与交付

原聚合函数在完整评测分母上生成报告，另加 partial_snapshot 标识和源目录；额外 module-metrics.json 仅展示本轮模块。judge 的 snapshot_complete 仅表示本次冻结集合完成，不表示整个 baseline 完成。截断回答保留并参与判分，MCQA 本地评分。给出 SCP、前台两条检查、后台继续及查看日志命令；不改动正在运行的源代码/依赖。

2026-10-18 用户回传视觉探针 `vision-judge-probe-20261018T041051.839029Z/report.json` 通过：两个不同排列控制图全部匹配，SIUO gen/1001 严格标签解析通过，服务报告 image_tokens。启动器 `scripts/judge_completed_server.sh` 复用此报告，不重复探针，前台正式判 MSSBench 与 SIUO 各一条后后台继续。本轮 API 预计 1,960 + 167 = 2,127 个回答，SIUO MCQA 167 条本地计分（空回答规则不调用 API）。

验证：`C:\anaconda3\python.exe -m unittest discover -s tests -p test_partial_judge.py -v`。必须覆盖活动生成锁仍在时只读快照、尾部半行/坏中间行、篡改与重复、错误接口及凭据脱敏、重试/复用、指标分母、实际 API 不发送图像。

最终验证：13 项新增离线测试通过，全套 179 项测试通过，启动器 Bash 语法检查与 git diff --check 通过。测试使用合成数据和模拟 API；SIUO 路径明确验证通过探针后保留图像输入，MSSBench 路径只发送文本。真实批量判分尚待用户执行，未将本地测试当作服务器已启动。
