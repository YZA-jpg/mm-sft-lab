# Smoke 与小批量推理验收

2026-10-15 更新：本页保留初始 greedy 检查流程。真实推理与 48 条采样试跑已完成，最新配置、结果和下一阶段见 [采样 baseline](sampling-baseline.md)。下面的环境问题属于首次准备时的记录。

此流程用于确认 Qwen3-VL 的图片、文本、padding 与回答对应正确，再进入全量 baseline。代码在本地编写；Git 同步和服务器操作由用户执行。当前尚未启动真实模型，也未调用 judge API。

本地已完成：83 项离线测试、Bash 语法检查、真实数据 12 / 12 / 48 选样和包装脚本 plan；mock 的三阶段生成/验收与 batch 对照均通过。服务器目前仅确认旧版 64 项离线测试通过，新代码须同步后重新验证。

## 固定的三个阶段

| 阶段 | 配置 | Batch | SIUO gen / MCQA | MSSBench chat / embodied | 总请求 / 配对 |
| --- | --- | ---: | --- | --- | --- |
| smoke | `configs/baseline_smoke.json` | 1 | 2 / 2 | 4 / 4 | 12 / 4 |
| batch 对照 | `configs/baseline_smoke_batch4.json` | 4 | 2 / 2 | 4 / 4 | 12 / 4 |
| small | `configs/baseline_small.json` | 4 | 8 / 8 | 16 / 16 | 48 / 16 |

MSSBench 的 `per_task_limit` 以完整 safe/unsafe 配对计数，SIUO 以问题计数。三个阶段都按稳定 ID 排序选择；small 包含 smoke 的选样。smoke 与 batch 对照使用完全相同的题目、图片、标签与 greedy 设置，仅 batch size 不同。默认 BF16、SDPA、1024 新 token、不增加 system prompt、不改变 processor 默认图片预算。

完整数据为 2,294 条单回答请求；小批量检查通过后，另用 `configs/baseline_full.json` 建立新 run，不能把 smoke/small 当作完整 baseline。

## 已知环境问题与当前边界

服务器已有 `.venv-data` 数据环境，可执行不加载模型的 `plan` 和离线单元测试。推理专用 `.venv` 尚未配置验证。

现有 `/data1/yaoziao/conda-envs/ipiguard-py310` 实测 Python 3.10.21、torch 2.5.1+cu124、Transformers 4.48.0，缺少 Pillow、accelerate 和 torchvision；导入 torch 报错：

```text
libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
```

该输出表明 CUDA 动态库解析失败，具体加载了哪份库仍需检查。不要据此直接在其他项目环境里升级或替换库。回来后先检查动态库路径与可用安装来源，再确定独立推理环境；目前不执行环境修复或模型启动。独立环境的待验证安装方案见 [baseline 文档](baseline.md)。

本地 `.venv-data\Scripts\python.exe` 在此次助手检查时无法找到其基础 Python；离线检查使用本机可运行的 Python 3.12.7。该现象与服务器数据环境是否可用无关。

## 1. 本地 PowerShell：验证并同步代码

使用本地可运行、已安装 Pillow 的 Python；无需 torch、GPU 或 API key。以下使用 `python`，执行前确认它指向可用解释器。

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"

python --version
if ($LASTEXITCODE -ne 0) { throw "Python 不可用" }
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "离线测试失败" }
git diff --check
if ($LASTEXITCODE -ne 0) { throw "补丁格式检查失败" }

git add README.md configs/baseline_smoke_batch4.json configs/baseline_small.json docs/baseline.md docs/workflow.md docs/smoke-tests.md docs/plans/2026-10-14-smoke-and-small-batch.md scripts/run_checks.sh src/mm_sft_lab/backends.py src/mm_sft_lab/runner.py src/mm_sft_lab/cli.py src/mm_sft_lab/diagnostics.py tests/test_baseline_framework.py tests/test_generation_checks.py
if ($LASTEXITCODE -ne 0) { throw "暂存失败" }
git --no-pager diff --cached --stat
if ($LASTEXITCODE -ne 0) { throw "查看暂存修改失败" }
git commit -m "Add smoke validation and small-batch checks"
if ($LASTEXITCODE -ne 0) { throw "提交失败" }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "推送失败" }
git status
```

## 2. 服务器 Bash：同步并核对选样，不加载模型

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  git pull --ff-only
  git --no-pager log -1 --oneline
  .venv-data/bin/python -m unittest discover -s tests -v
  PYTHON_BIN="$PWD/.venv-data/bin/python" bash scripts/run_checks.sh plan
)
```

预期三个配置分别选中 12、12、48 条请求；前两者 `selection_sha256` 均为：

```text
4f2fe211ebde6abba52a1ba0da090da0021bdb86201e8557ba689da9b35ef131
```

数据根默认是服务器 `data/imported`。不要重新 prepare 覆盖传来的 `prepared`；后续本地 judge 需要与服务器相同的数据快照。运行日志保存到 `runs/qwen3-vl-4b-checks/logs/`，不会写入待生成的 run 目录。

## 3. 环境就绪后：分阶段启动

下面三个块分别执行，每个阶段查看结果后再继续。推理环境必须能够导入 torch 和 Qwen3-VL，支持单卡 CUDA/BF16；框架不会代为安装依赖。每个阶段都重新加载模型，因此可以分别安排运行。

首先查询 GPU 7 的空闲显存；用户此前给出 48,700 MiB 空闲，实际运行前以新查询为准。

```bash
nvidia-smi -i 7 --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
```

12 条 batch 1：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  CUDA_VISIBLE_DEVICES=7 bash scripts/run_checks.sh smoke
)
```

同一 12 条 batch 4，自动比较输出：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  CUDA_VISIBLE_DEVICES=7 bash scripts/run_checks.sh batch
)
```

48 条 batch 4：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  CUDA_VISIBLE_DEVICES=7 bash scripts/run_checks.sh small
)
```

脚本默认 `PYTHON_BIN=$PWD/.venv/bin/python`、`DATA_ROOT=$PWD/data/imported`、`RUN_ROOT=$PWD/runs/qwen3-vl-4b-checks`；可通过同名环境变量显式覆盖。强制本地模型加载的 HF/Transformers offline 模式。代码本身不会连接 judge API。

需要补完同一阶段时，例如：

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_checks.sh smoke --resume --retry-errors
```

`--resume` 仅补缺失项；加 `--retry-errors` 也重试已记录失败项。成功项不会重复生成。配置、数据、模型指纹、代码或依赖发生变化时，请使用新的 `RUN_ROOT`，避免混合实验。若旧 `.generate.lock` 遗留，先核对文件中的 host/PID 并确认进程结束，再人工处理锁。

## 验收规则与输出

`validate-run` 的退出码：0 表示技术检查 `passed`，2 表示 `needs_review` 或 `failed`，1 表示输入/文件等执行异常。脚本遇到非零退出即停止后续步骤。

- `failed`：生成缺失/失败、空回答、图片 grid 不匹配、padding 不一致、token 数或请求身份异常。
- `needs_review`：回答截断、终止原因异常、MCQA 无法严格解析、缺失测量或出现 OOM 拆批。若进程在追加 batch 日志期间中断，部分 batch 测量也会待复核，不计入吞吐统计。
- `passed`：上述检查通过，仍须查看原始回答确认图文对应；它不代表安全与能力评测通过。

`compare-batches` 要求两个 run 的数据/模型/代码/环境/生成设置一致，仅允许 `batch_size` 不同。文本、输入/输出 token 数、图片 grid、MCQA 解析结果或终止原因变化都会列出供复核；另区分只改变空白与实际内容变化。greedy 在不同批大小下仍可能受数值计算影响，差异不自动等于实现错误。比较还要求真正执行过配置大小的 batch；OOM 后全部拆成单条不能通过 batch 4 检查。任一 run 未通过验收时比较也不会通过。

`small` 开始前会根据当前 journal 重新计算 batch 对照，避免使用过期的比较文件。检查有待复核项时先保留输出分析，不自动删回答、降低图片预算或切换生成参数。

```text
runs/qwen3-vl-4b-checks/
  logs/
  smoke-b1/
    manifest.json / selection.jsonl / evaluation.jsonl / predictions.jsonl
    checks/validation.json / validation.md / review.jsonl / review.csv
  smoke-b4/
    ...
  batch-comparison/
    comparison.json / comparison.md / review.jsonl / review.csv
  small-b4/
    ...
```

逐回答记录图片数量、`image_grid_thw`、有效输入 token、padding 宽度、输出 token、终止原因、实际 batch 大小、batch ID、计时和 CUDA allocated/reserved 峰值。grid 与图片的对应依据 [Qwen3-VL processor 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)；显存统计依据 [PyTorch CUDA memory API 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py)。峰值包含已驻留的模型与 allocator cache，不能等同于整张卡的占用或其他用户的进程占用。

吞吐按成功 batch 的 wall time 汇总，每个 batch 只计时一次；包含该次预处理、生成、解码，排除模型加载、失败/OOM 尝试和 journal 写入。仅作本次 smoke 的描述性记录，不能作为稳定吞吐 benchmark。所有 mock 结果明确标记，`real_local_generation_passed` / `real_local_batch_check_passed` 对 mock 恒为 false。

也可单独离线复核已有 run，无需加载模型：

```bash
.venv-data/bin/python scripts/lab.py validate-run --run-dir runs/qwen3-vl-4b-checks/smoke-b1
.venv-data/bin/python scripts/lab.py compare-batches \
  --before runs/qwen3-vl-4b-checks/smoke-b1 \
  --after runs/qwen3-vl-4b-checks/smoke-b4 \
  --output runs/qwen3-vl-4b-checks/batch-comparison
```

后续安全性与有效性判分继续使用独立的 `judge → score`；若服务器无法连接 judge 服务，可将 run 产物拉回本地并指定本地数据根。具体命令等 smoke 回答与 API 服务配置确认后再执行。
