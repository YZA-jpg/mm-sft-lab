# B0 baseline 收尾 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 复用已完成的判分，只补 WildJailbreak，并生成含分母和覆盖率的 B0 完整主表。

**Architecture:** 保留现有 v3 judge 工具及语义协议，在新的完整主生成快照上导入 2,124 条有效标签和三条平台缺失，仅新增 2,210 条 WildJailbreak 判分。独立只读汇总器重新聚合真实 journal、校验报告身份，输出 JSON、CSV 和 Markdown；不修改 src、环境或旧 judge 文件。

**Tech Stack:** Bash、Python 标准库、既有 mm_sft_lab 数据/聚合函数；离线合成数据与 mock API。

---

## 已确认范围

- 用户选择保留 SIUO 三条缺失，主结果 89/164（54.2683%），coverage 164/167（98.2036%）。不进行人工补评或切换 judge。
- 用户选择本轮以 TextVQA Val 为能力主指标；Test 5,734 条保留待官方评分，不用 Val 分数填入 Test 栏。
- B0 主生成 9,504 条已完成；新增 WildJailbreak harmful 2,000＋benign 210，分别报告 ASR 和 overrefusal。
- 保持服务器 v3 工具 SHA256 `5831d296fe06b9d4bd57622f5f0d3d322a9faa904fe352e163ce84b99afe811b`，不重新判成功样本，不重试已知平台拒绝。
- 用户执行上传、服务器命令、API 调用与 Git 操作；本地仅修改工具、文档及离线验证。当前授权优先于技能中的新 worktree、自动提交与重复确认建议。

## Task 1: 验证跨完整快照复用

**Files:** 新建 `tests/test_b0_completion.py`；复用 `scripts/judge_partial.py`，保持其字节不变。

1. 合成同一 fingerprint 的早期局部快照与完整快照。
2. 保留一条平台拒绝；验证新快照导入成功与拒绝记录，plan 不调用 API。
3. 验证新增调用均为 WildJailbreak、旧目录字节不变、再次 resume 不重复请求。

## Task 2: 准备用户执行的收尾启动器

**Files:** 新建 `scripts/complete_b0_server.sh`。

1. start 校验完整生成、旧判分计数、v3 字节及启动锁。
2. 冻结全量 main；plan 预期语义目标 4,337、复用成功 2,124、平台缺失 3、新请求 2,210。
3. 隐藏输入 Key；前台两条检查后后台继续，使用独立 `latest-b0-completion.txt` 指针，保留旧指针。
4. resume 复用同一快照与成功行；status 不调用 API。worker 结束后自动导出表格。

## Task 3: 生成可审计的完整主表

**Files:** 新建 `scripts/report_b0.py` 与相应离线测试。

1. 用原 selection/evaluation/predictions/judgments 重算指标，核对 fingerprint、签名基础、原始 judge 标签和已保存 report。
2. 分别输出 SIUO gen、MSSBench chat/embodied、Wild harmful/benign、SIUO MCQA、TextVQA Val，记录分子、有效分母、全量分母、截断和缺失。
3. 仅允许用户确认的三条 SIUO 缺失作为主表收尾例外；Wild 不完整时不宣称主表收尾。
4. Test 一律保留 pending_official；mock 表显式标记，不能称真实 baseline。

## Task 4: 文档与交付

**Files:** 新建 `docs/b0-completion.md`；更新 `docs/data-and-experiment-notes.md`、`docs/b0-progress-2026-10-19.md`。

说明当前授权选择、上传/启动/查看/恢复/取回表格的分阶段命令。离线定向测试、全套回归、Bash/Python 语法与 git diff 检查通过后交付。没有真实 Wild 分数回传前不预填数值，不声称后台启动或 API 完成。

## 验证与待执行状态

2026-10-20 实现完毕：四项新增合成测试及全套 193 项离线测试通过；Bash/嵌入 Python 语法、diff 检查通过，v3 引擎字节未改。用户已确认本轮采用 TextVQA Val、保留 SIUO 三条缺失。真实服务器启动、WildJailbreak 2,210 条 API 判分、主表真实导出与回传尚待用户执行；未自动提交 Git。
