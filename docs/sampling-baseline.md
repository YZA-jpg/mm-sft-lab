# 采样 baseline 的已验证配置与阶段记录

> 文档状态（2026-10-23）：历史采样实验记录，保留 1024/2048 token 对照。当前执行入口见 [baseline 手册](baseline-runbook.md)。

更新日期：2026-10-15（北京时间）。本页依据用户回传的服务器结果记录；原始 run 文件保存在服务器，尚未导入本地。全量 baseline、安全/有效性判分和 SFT 尚未完成。

## 当前生成配置

48 条试跑使用的 `sampling_2048_v1` 已整理为仓库配置文件。运行参数为本地 Qwen3-VL-4B-Instruct、单卡 GPU 7、BF16、SDPA、batch 1、2048 新 token、temperature 0.7、top_p 0.8、top_k 20、一个基础 seed 42。每题实际 seed 由基础 seed、ID 和回答索引派生，保存在 predictions.jsonl。

| 配置 | SIUO gen / MCQA 请求数 | MSSBench chat / embodied 请求数 | 总请求 / MSSBench 配对 |
| --- | --- | --- | --- |
| `configs/baseline_sampling_smoke.json` | 2 / 2 | 4 / 4 | 12 / 4 |
| `configs/baseline_sampling_small.json` | 8 / 8 | 16 / 16 | 48 / 16 |
| `configs/baseline_sampling_full.json` | 167 / 167 | 1200 / 760 | 2294 / 980 |

三个配置仅选样上限不同；MSSBench 按完整配对选择。48 条按稳定 ID 选样，覆盖 SIUO gen / MCQA 各 8 个 source scene、MSSBench chat 4 个、embodied 2 个，用于运行稳定性检查，不能外推全量分数或截断率。

原 `baseline_smoke.json`、`baseline_smoke_batch4.json`、`baseline_small.json`、`baseline_full.json` 保留 `controlled_v1` 的 greedy 配置。`scripts/run_checks.sh` 仍运行原 greedy 流程；采样配置使用 `scripts/lab.py` 的明确配置路径。`siuo_3sample.json` 是另一个生成协议，不能与单次采样结果混表。

SFT 前后应固定生成协议、选样、token 预算、batch、依赖和判分规则。任何改变都建立新的 run；已有回答及其指纹保留。

## 服务器验证记录

代码提交 `fb33633` 已同步服务器，83 项离线测试通过。项目推理环境 `.venv` 已能运行模型：Python 3.11、torch 2.8.0+cu128、torchvision 0.23.0、Transformers 4.57.1；其他版本以 run manifest 为准。之前 `ipiguard-py310` 的 CUDA 导入错误属于另一个环境。

推理命令在子 shell 中清理继承的 `LD_LIBRARY_PATH`、`LD_PRELOAD`、`PYTHONPATH`、`PYTHONHOME`，设置 `PYTHONNOUSERSITE=1`、`CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `CUDA_VISIBLE_DEVICES=7`，使用指定 `.venv/bin/python`。这些设置用于保持已验证运行方式，不修改用户全局 shell 配置。

| 阶段 | 结果 | 服务器 run（相对仓库根目录） |
| --- | --- | --- |
| greedy、batch 1、1024 | 12/12 成功，3 条截断 | `runs/qwen3-vl-4b-checks-20261014T163345Z/smoke-b1` |
| sampling、batch 1、1024 | 12/12 成功，1 条截断，成功 batch 总耗时 126.85583 秒 | `runs/qwen3-vl-4b-sampling-smoke-20261014T173201Z/run` |
| sampling、batch 1、2048 | 12/12 成功，同一条仍截断，其余 11 条回答完全相同，耗时 165.127177 秒 | `runs/qwen3-vl-4b-length2048-smoke-20261014T174511Z/run` |
| greedy、batch 4、1024 | 12/12 成功，实际 3 个 batch 4，8 条文本与 batch 1 不同 | `runs/qwen3-vl-4b-batch4-smoke-20261014T175207Z/run` |
| 48 条 sampling、batch 1、2048 | 48/48 成功，无错误/缺失，16 个配对完整，3 条截断，重合 12 条全部复现 | `runs/qwen3-vl-4b-small48-20261015T025151Z/run` |

1024→2048 的对照中，唯一触顶回答继续重复推演；成功 batch 总耗时增加约 30.2%。两轮记录的显存 allocated 峰值均四舍五入为 8.76 GiB，这不证明其他样本或 batch 下增加长度没有显存成本。12 条结果不足以证明全量所需预算。

batch 对照后，在 CPU 重新逐条核对 processor：去 padding 的 token、图片 grid、像素张量在 12 条中全部逐位一致，像素最大绝对差为 0。报告在 `runs/qwen3-vl-4b-batch4-smoke-20261014T175207Z/comparison/input-alignment.json`。模型内部数值或 padding 处理的具体差异尚未定位；当前继续固定 batch 1。

48 条成功 batch 总耗时 615.07339 秒，3 条截断均在 MSSBench embodied：

- `mssbench/embodied/0000/1/a`
- `mssbench/embodied/0001/1/b`
- `mssbench/embodied/0001/2/b`

这是总体 3/48、该任务 3/16 的截断统计，不是安全或能力分数。`validate-run` 的 `needs_review` 保留输出质量问题；完整且可核对的截断回答参与后续原协议评分，不丢弃或自动记为失败。

## 当前下一阶段：先做无需 API 的统计

服务器 Bash：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  .venv-data/bin/python scripts/lab.py score \
    --run-dir runs/qwen3-vl-4b-small48-20261015T025151Z/run
)
```

输出在该 run 的 `unjudged_report/`。MCQA 的 8 条回答直接与正确选项对比；其余 40 条的安全/有效性分数显示 pending/null。`complete: false` 表示评分尚未完成，不表示生成失败。

判分方式可采用人工、本地模型或远程 API。现有真实自动 judge 入口为兼容 Chat Completions API；本地模型也可提供本地接口，纯 Python 本地 judge 或人工标签导入尚未实现，需要选择后再增加对应入口。`--mock` 只生成模拟标签用于测试流程。

SIUO generation 的语义判分需要结合图片、题目、warning 与参考答案；MSSBench 当前采用回答文本的 0–3 分类。现有 `lab_rubric_v1` 为项目简明 rubric，未逐字复现官方 judge prompt。未来确定判分方式后，先做少量真实判分检查，再评测 40 条开放回答；之后冻结完整评测协议并启动全量生成。

本页中的新配置与记录由助手在本地整理，Git 提交、推送、服务器同步和每个测试阶段仍由用户操作。环境检测、推理测试、判分测试与全量运行分别结束反馈，不自动串联跨阶段。
