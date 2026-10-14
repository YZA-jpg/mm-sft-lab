# 本地修改与服务器测试流程

## 工作约定

- 本地项目目录：`D:\Desktop\AI Safety\mm-sft-lab`，直接在该目录工作。
- 远程仓库：`git@github.com:YZA-jpg/mm-sft-lab.git`。
- 采用 main 分支进行初期同步；首次提交 `81b1a09` 已推送到 origin/main。
- 助手修改本地项目文件，并说明本次改动、检查结果和下一步操作。
- Git 提交、推送、服务器拉取、数据下载和服务器测试，由用户执行给出的命令。
- 每轮命令明确标注本地 PowerShell 或服务器 Bash，提供预期结果；依赖前一步结果的操作等用户反馈后继续。
- 命令包含失败停止检查，报错时保留完整输出后排查。

## 实验环境

- 第一阶段使用一张 NVIDIA A100-SXM4-80GB；用户提供的物理 GPU 编号为 7。
- 模型路径：`/data1/sharehub/models/Qwen3-VL-4B-Instruct`。
- 服务器仓库路径：`/data1/yaoziao/mm-sft-lab`；已校验的离线数据目录为其下的 `data/imported/`。
- 服务器 Git 2.34.1，base Python 3.14.7，Conda 26.7.1，驱动 580.159.03；数据环境 `.venv-data` 已通过 baseline 框架原有 64 项测试。
- 默认 Conda 频道遇到 ToS 检查，创建项目环境时使用 `--override-channels --channel conda-forge`。
- 已有 `ipiguard-py310` 导入 torch 时出现 `__nvJitLinkComplete_12_4` 动态库错误，Transformers 4.48.0 亦低于 Qwen3-VL 要求；GPU 推理环境与真实 judge 服务仍待配置。
- 下一轮按 [smoke 与小批量步骤](smoke-tests.md) 同步新代码，先核对选样，再处理独立推理环境并分阶段启动。

## 推进顺序

1. 完成本地 Git 初始化、首次提交与推送。
2. 本地增加数据下载、规范化和完整性校验脚本。
3. 用户推送该批修改，在服务器克隆或拉取同一提交。
4. 本地下载与校验 SIUO、MSSBench，通过 SCP 上传离线包，在服务器解压复核。
5. 本地增加模型推理与评测入口，服务器依次执行 12 条 smoke、同清单 batch 1/4 对照、48 条小批量测试。
6. 小样本通过后运行全量 baseline，保留回答、评分和运行配置。

## 数据与产物

- 大体积数据、官方 benchmark 工作副本、模型权重、checkpoint、运行产物和本地密钥不提交 Git。
- 数据下载到服务器指定目录，仓库保存下载代码、来源、revision 和校验信息。
- 服务器代码更新使用 `git pull --ff-only`；若出现本地修改或分支分叉，先检查原因。
- 每次服务器测试记录 `git rev-parse HEAD`，用于对应本地代码版本。

当前进度：baseline 框架提交 `124c9a8` 已推送并同步服务器，原有 64 项测试在服务器通过；离线数据包上传及复核通过。2,294 条请求、980 个 MSSBench 配对、919 个图片文件，两端一致。smoke 与小批量检查代码在本地继续完善，Git 与服务器操作由用户回来后执行，尚未启动模型或调用真实 judge API。
