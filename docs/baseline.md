# Baseline 框架与服务器启动流程

当前已实现可执行框架，并通过离线测试；真实 Qwen 模型加载、GPU 推理和在线 judge 仍待服务器验证。用户稍后回来时按下面阶段逐步执行并反馈输出。

本轮新增的分阶段操作以 [smoke 与小批量说明](smoke-tests.md) 为准：12 条 batch 1、同清单 batch 4 对照、48 条小批量。现有 `ipiguard-py310` 的 torch 导入失败，推理环境就绪前只执行选样和离线检查。

## 入口与目录

无需安装本仓库也可在仓库根目录运行 `python scripts/lab.py --help`；执行 editable install 后也能用 `mm-sft-lab` 或 `python -m mm_sft_lab`。

| 命令 | 职责 |
| --- | --- |
| `preflight` | 检查本地 checkpoint、依赖和单卡 CUDA；API 模式检查配置，不发推理请求 |
| `plan` | 核对 prepared 文件与图片哈希、打印固定小样本清单的统计，不加载模型 |
| `generate` | 保存原始回答、token 数、终止原因、错误；支持续跑 |
| `validate-run` | 检查生成完整性、图文输入、padding、截断和 batch 测量；不调用 judge |
| `compare-batches` | 比较同一清单仅改变 batch size 的 greedy 回答与输入 grid |
| `judge` | 使用独立 API 判分；SIUO 传图片，MSSBench 使用回答文本分类 |
| `score` | 输出 coverage、任务/类别指标、CSV 和 Markdown 报告 |
| `compare` | 对同数据、同生成协议、同 judge 的完整结果输出差值与 improved/regressed ID |

```text
src/mm_sft_lab/
  config.py        # 严格配置验证
  data.py          # prepared 完整性与完整配对选择
  benchmarks.py    # benchmark prompt 与 MCQA 解析
  preflight.py     # 本地 checkpoint 和环境记录
  backends.py      # Transformers / Chat Completions API / mock
  runner.py        # 生成、运行指纹与断点恢复
  diagnostics.py   # smoke 验收、batch 对照与人工复核清单
  judge.py         # 独立 API judge、原始判分与标签
  metrics.py       # 聚合、报告和逐样本比较
  cli.py           # 命令行接口
```

训练配置约定放在 `configs/train/README.md`，实际 SFT 入口将在 baseline 冻结后实现。

## 本地离线检查

在本地 PowerShell、仓库根目录执行：

```powershell
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "测试失败" }
python scripts/lab.py --help
```

可用新目录演示全链路，demo 与 mock 都有明确标记。以下命令同样可在服务器执行；每行成功后再继续：

```text
python scripts/make_demo.py --data-root data/demo
python scripts/lab.py plan --config configs/mock.json --data-root data/demo
python scripts/lab.py generate --config configs/mock.json --data-root data/demo --run-dir runs/demo
python scripts/lab.py generate --config configs/mock.json --data-root data/demo --run-dir runs/demo --resume
python scripts/lab.py judge --run-dir runs/demo --mock
python scripts/lab.py score --run-dir runs/demo --judge-dir runs/demo/judge
```

预期：6 条生成请求、5 条需 judge 的回答；续跑不重复成功项；`runs/demo/judge/report.md` 显示 `SIMULATION ONLY`。模拟分数不能作为模型表现。demo 数据目录须为空，重做时选新目录。

## 服务器已知状态

- 仓库：`/data1/yaoziao/mm-sft-lab`；已确认服务器同步到 baseline 框架提交 `124c9a8`。
- 本地模型：`/data1/sharehub/models/Qwen3-VL-4B-Instruct`，目录及 config 可访问。
- 物理 GPU 7：A100-SXM4-80GB；最新一次用户检查为 48,700 MiB 空闲，运行前重新查询。
- 服务器 base 的 Python 是 3.14；数据环境使用仓库内 `.venv-data`，原有 64 项测试已通过。
- 已有 `ipiguard-py310` 为 Python 3.10.21、torch 2.5.1+cu124、Transformers 4.48.0，导入 torch 报 `__nvJitLinkComplete_12_4` 动态库错误；目前不修改该环境，独立推理环境尚待验证。
- 服务器直接下载曾因 `[Errno 110] Connection timed out` 失败；已在本地完成下载与校验，并通过 `tstar_001` 上传约 441 MB 的离线包，服务器 SHA-256 校验通过。
- 服务器数据目录：`/data1/yaoziao/mm-sft-lab/data/imported`；离线图片/标注/配对复核通过，统计与本地一致。后续评测命令使用 `--data-root data/imported`。

## 第一步：同步这一批代码并检查选样

用户先在本地提交推送，再于服务器 Bash 执行：

```bash
(
  set -e
  cd /data1/yaoziao/mm-sft-lab
  git pull --ff-only
  git log -1 --oneline
  .venv-data/bin/python -m unittest discover -s tests -v
  .venv-data/bin/python scripts/lab.py plan --config configs/baseline_smoke.json --data-root data/imported
)
```

当前服务器数据已完成导入，复核结果保存在 `data/imported/server-check`，评测使用传来的原始 `data/imported/prepared` 快照。无需再次下载或覆盖 prepared；数据来源与统计见 [数据快照](data-snapshot.md)。

smoke 按排序后的稳定 ID 选择 SIUO 每个任务前 2 题、MSSBench 每个任务前 2 个完整 query/instruction 配对，通常合计 12 条请求；实际不足 2 个单位时以清单为准。选择文件将冻结在运行目录中。

## 第二步：创建 GPU / API 实验环境

为推理创建独立 `.venv`，保留数据环境。先在服务器 Bash 检查 GPU：

```bash
nvidia-smi -i 7 --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
```

下面给出待服务器验证的安装组合；CUDA 12.8 的 PyTorch wheel 来自 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)，Qwen3-VL 需要 Transformers 4.57 及以上，见 [Qwen 官方说明](https://github.com/QwenLM/Qwen3-VL#quickstart)。当前框架限制在 Transformers 4.x，以减少主版本变化。

```bash
(
  set -e
  cd /data1/yaoziao/mm-sft-lab
  if [ -e .venv ]; then
    echo '.venv 已存在，请先确认环境状态'
    exit 1
  fi
  conda create --prefix "$PWD/.venv" --override-channels --channel conda-forge python=3.11 pip -y
  .venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install -r requirements-inference.txt -r requirements-api.txt
  .venv/bin/python -m pip install --no-deps -e .
  .venv/bin/python -m pip check
  .venv/bin/python -m unittest discover -s tests -v
  mkdir -p configs/local
  .venv/bin/python -m pip freeze > configs/local/runtime.freeze.txt
)
```

此环境组合尚未实际安装验证。smoke 通过后保存 freeze 文件；后续实验复用明确版本。只走 API 时可创建同样的 Python 环境后只安装 `requirements-data.txt` 与 `requirements-api.txt`，不需要 CUDA 依赖。

## 第三步：本地 Qwen 小样本推理

推理环境就绪后，在服务器 Bash 执行第一阶段；脚本先做 preflight，再 plan、generate、validate-run：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  CUDA_VISIBLE_DEVICES=7 bash scripts/run_checks.sh smoke
)
```

模型和 processor 仅从指定本地目录加载。进程内使用逻辑 `cuda:0`，要求恰好暴露一张 GPU。默认 BF16、SDPA、batch 1、greedy、1024 新 token；不修改 processor 默认视觉预算，不追加安全 system prompt。批量推理左侧 padding，按完整 padding 宽度裁去输入 token。

smoke 回答检查通过后，执行 `bash scripts/run_checks.sh batch` 比较同一清单 batch 1/4 的回答；对照通过后执行 `bash scripts/run_checks.sh small` 跑 48 条。每个阶段分别执行，产物与复核规则见 [smoke 与小批量说明](smoke-tests.md)。确认单样本/批量的图片、文本与回答对应后，再用 `configs/baseline_full.json` 跑全量。SIUO 三采样使用单独的 `configs/siuo_3sample.json`，不能与 controlled_v1 混表。

## 模型 API 入口

客户端使用 [Chat Completions 多模态格式](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) 与 Base64 图片。兼容服务需支持图片输入及配置中的参数。服务端部署不属于本次客户端实现；如果本地 Qwen 以 API 服务启动，稍后一起确定服务后端和推理参数。

在服务器当前 shell 中配置实际服务信息；不要把密钥写入 JSON 或提交 Git：

```bash
export MODEL_BASE_URL='https://YOUR_MODEL_HOST/v1'
export MODEL_ID='YOUR_SERVED_MODEL_ID'
read -rsp 'Model API key: ' MODEL_API_KEY
echo
export MODEL_API_KEY
```

然后执行模型 API smoke：

```bash
(
  set -e
  cd /data1/yaoziao/mm-sft-lab
  .venv/bin/python scripts/lab.py preflight --config configs/model_api.example.json
  .venv/bin/python scripts/lab.py generate --config configs/model_api.example.json --data-root data/imported --run-dir runs/model-api-smoke
)
```

`base_url` 是服务根地址（通常以 `/v1` 结尾），不是完整 `/chat/completions` 路径。示例采用 temperature 0，协议 ID 与本地 greedy 分开；API 服务不能仅凭模型名保证与服务器 checkpoint、processor 和数值实现一致。

API 全量运行时将示例复制到 `configs/local/`，把 `per_task_limit` 设为 `null`，使用新 run 目录；示例本身只选择 smoke 样本。

## 第四步：独立 API judge

judge 可以使用另一服务和模型。SIUO generation 需要视觉模型；MSSBench 采用回答文本分类。同一轮 SFT 前后应固定 judge 模型/snapshot、端点和配置。

```bash
export JUDGE_BASE_URL='https://YOUR_JUDGE_HOST/v1'
export JUDGE_MODEL='YOUR_VISION_CAPABLE_JUDGE_ID'
read -rsp 'Judge API key: ' JUDGE_API_KEY
echo
export JUDGE_API_KEY
```

服务信息就绪后，服务器 Bash 执行：

```bash
(
  set -e
  cd /data1/yaoziao/mm-sft-lab
  .venv/bin/python scripts/lab.py judge --config configs/judge.example.json --run-dir runs/qwen3-vl-4b-checks/smoke-b1
  .venv/bin/python scripts/lab.py score --run-dir runs/qwen3-vl-4b-checks/smoke-b1 --judge-dir runs/qwen3-vl-4b-checks/smoke-b1/judge
)
```

`lab_rubric_v1` 使用本项目的简明 JSON rubric，保留 SIUO 的安全/三档有效性语义、MSSBench 0–3 类与聚合规则；并非官方 judge prompt 的逐字复现。依据：[SIUO 评分代码](https://github.com/sinwang20/SIUO/blob/main/eval/gpt-eval.py)、[MSSBench 评分代码](https://github.com/UCSB-AI/MSSBench/blob/main/utils/gpt4_eval.py)。更换 judge 时指定新的 `--judge-dir`，报告分别存放。

API 默认传 `max_tokens`。若服务需要 `max_completion_tokens`，在忽略 Git 的 `configs/local/` 中复制并修改配置，再建立新 run/judge 目录。其他不兼容参数会明确报错，不自动删参数后重试。seed 默认不发给 API，不能宣称服务端确定性；需要且支持时显式配置 `send_seed`。

## 恢复、产物与统计口径

生成恢复使用原配置与 run 目录加 `--resume`；同时补跑已记录失败项则加 `--retry-errors`。judge 同理。成功项不会重复调用，错误尝试保留；仅截断恢复日志末尾未完成的 JSON 行，文件中间损坏会报错。默认 API 最多重试 2 次；连续 3 个批次/判分失败会停止，400/401/403/404 立即停下供排查。

代码、配置、数据、依赖版本或模型指纹变化时拒绝混入旧 run。模型指纹包含配置文件哈希与权重文件名/大小/mtime 清单，未完整 hash 大型权重；模型目录应保持不变。若需要强权重校验，可另行保存权重 SHA-256 清单。API 则保存端点、请求模型名和返回的模型/指纹信息。

OOM 会拆小当前 greedy batch 重试；batch 1 仍失败会记录错误，不降低图片分辨率。采样只允许 batch 1，实际请求种子由固定基础 seed、样本 ID 与回答索引派生，避免中断续跑改变 RNG 顺序。不同硬件/实现仍可能产生数值差异。

运行产物：

```text
runs/<run>/
  manifest.json
  resolved_config.json
  selection.jsonl
  evaluation.jsonl
  predictions.jsonl
  errors.jsonl                 # 存在生成错误时
  checks/                     # validate-run 生成，无需 API
    validation.json
    validation.md
    review.jsonl
    review.csv
  judge/
    manifest.json
    judgments.jsonl
    metrics.json
    summary.csv
    report.md
```

`score` 不带 `--judge-dir` 时也可运行，MCQA 可直接统计，其余任务显示 pending/null，报告写入 `unjudged_report/`。未完成 coverage 时不能作为完整 baseline。

SIUO 三回答先在每条回答计算安全且有效，再对问题多数聚合；类别分组保留相同逻辑。MSSBench 分别报告 safe/unsafe/total、完整配对同时正确率、0–3 类分布和安全场景能力失败率。安全场景 class 3 在官方口径中算正确，因此 safe accuracy 不能当成任务完成率。空回答和截断单列统计。

`compare --before <metrics.json> --after <metrics.json> --output <comparison.json>` 要求两个非 mock 的完整报告、相同数据清单、生成设置、代码与 judge 协议。当前输出为描述性差值和 improved/regressed ID，不含 bootstrap 置信区间。

运行目录有阶段锁防止并发写入。若进程被强制终止留下 `.generate.lock` / `.judge.lock`，先查看文件中的 host/PID 并确认对应进程已停止，再人工移除该锁；正常退出和 Ctrl-C 会释放锁。

MSSBench IF prompt 前缀保留上游文本和 MIT 版权许可，见 `src/mm_sft_lab/MSSBench-LICENSE.txt`。SIUO MCQA 使用上游简短模板；judge rubric 为本项目适配实现。
