# 本地修改与服务器测试流程

新增 Step 1 数据与训练入口的阶段命令见 [服务器执行手册](server-runbook.md)。当前训练入口覆盖 TextVQA、HH harmless chosen、VLGuard，评测加入 WildJailbreak eval 和 TextVQA；VLGuard 使用独立训练子根，保留现有父评测快照。数据处理和实验约定另见 [独立记录](data-and-experiment-notes.md)。新代码先完成本地离线验收，再由用户同步服务器。数据下载、环境检测、token 审计、训练 smoke、完整 baseline 与正式训练各自独立执行反馈。

## 工作约定

- 本地项目目录：`D:\Desktop\AI Safety\mm-sft-lab`，直接在该目录工作。
- 远程仓库：`git@github.com:YZA-jpg/mm-sft-lab.git`。
- 采用 main 分支进行初期同步；首次提交 `81b1a09` 已推送到 origin/main。
- 助手修改本地项目文件，并说明本次改动、检查结果和下一步操作。
- Git 提交、推送、服务器拉取、数据下载和服务器测试，由用户执行给出的命令。
- 每轮命令明确标注本地 PowerShell 或服务器 Bash，提供预期结果；依赖前一步结果的操作等用户反馈后继续。
- 每轮围绕一个阶段组织命令，阶段内可连续执行；环境检测、环境安装、smoke、判分和全量运行分别结束反馈，避免自动串联跨阶段。
- 命令包含失败停止检查，报错时保留完整输出后排查。

## 实验环境

- 第一阶段使用一张 NVIDIA A100-SXM4-80GB；用户提供的物理 GPU 编号为 7。
- 模型路径：`/data1/sharehub/models/Qwen3-VL-4B-Instruct`。
- 服务器仓库路径：`/data1/yaoziao/mm-sft-lab`；已校验的离线数据目录为其下的 `data/imported/`。
- 服务器 Git 2.34.1，base Python 3.14.7，Conda 26.7.1，驱动 580.159.03；数据环境 `.venv-data` 已通过 83 项离线测试。
- 默认 Conda 频道遇到 ToS 检查，创建项目环境时使用 `--override-channels --channel conda-forge`。
- `ipiguard-py310` 曾出现 CUDA 动态库导入错误；独立 `.venv` 已跑通真实推理，使用 Python 3.11、torch 2.8.0+cu128、torchvision 0.23.0、Transformers 4.57.1。
- 下一阶段先做本地规则统计，再确定人工、本地模型或 API 判分方式；见 [采样 baseline 记录](sampling-baseline.md)。

## 推进顺序

1. 完成本地 Git 初始化、首次提交与推送。
2. 本地增加数据下载、规范化和完整性校验脚本。
3. 用户推送该批修改，在服务器克隆或拉取同一提交。
4. 本地下载与校验 SIUO、MSSBench，通过 SCP 上传离线包，在服务器解压复核。
5. 本地增加模型推理与评测入口，服务器依次执行 12 条 smoke、同清单 batch 1/4 对照、48 条小批量测试。
6. 小样本稳定后检查评分链路，冻结协议，再分别启动全量生成和判分，保留回答、评分和运行配置。

## 数据与产物

- 大体积数据、官方 benchmark 工作副本、模型权重、checkpoint、运行产物和本地密钥不提交 Git。
- 数据下载到服务器指定目录，仓库保存下载代码、来源、revision 和校验信息。
- 服务器代码更新使用 `git pull --ff-only`；若出现本地修改或分支分叉，先检查原因。
- 每次服务器测试记录 `git rev-parse HEAD`，用于对应本地代码版本。

当前进度：框架提交 `fb33633` 已同步服务器，83 项离线测试通过；真实 GPU smoke、长度对照与 batch 输入对齐检查完成。48 条采样试跑全部生成成功，16 个配对完整，3 条 embodied 截断，重合的 12 条回答完全复现。全量数据仍为 2,294 条请求、980 个 MSSBench 配对、919 个图片文件。真实 judge、全量评测与 SFT 尚未完成。
