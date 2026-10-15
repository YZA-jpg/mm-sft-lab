# Step 1 分阶段执行手册

适用于本地 Windows `D:\Desktop\AI Safety\mm-sft-lab` 与服务器 `/data1/yaoziao/mm-sft-lab`，SSH 别名 `tstar_001`。每个阶段执行并反馈后再进入下一阶段，不能整篇一次粘贴运行。数据规则和实验限制见 [独立记录](data-and-experiment-notes.md)。

助手已完成本地代码与离线测试；以下真实准备、Git、传输、环境配置和模型运行由用户执行。当前旧服务器为 `fb33633`，需要先同步新代码。

## A. 本地：准备第三组并检查数据计划

保留现有父快照，创建 `data/step1/vlguard`。此阶段不启动模型。

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
$python = ".\.venv-step1\Scripts\python.exe"
& $python scripts/prepare_vlguard.py --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "VLGuard 准备失败" }
& $python scripts/lab.py plan --config configs/step1_smoke.json --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "评测计划失败" }
foreach ($dataset in @("textvqa", "hh", "vlguard")) {
    $trainingRoot = if ($dataset -eq "vlguard") { "data/step1/vlguard" } else { "data/step1" }
    & $python scripts/train.py plan --config "configs/train/${dataset}_lora.json" --data-root $trainingRoot
    if ($LASTEXITCODE -ne 0) { throw "$dataset 训练计划失败" }
}
$report = "runs/data-audit-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "/report.json"
& $python scripts/audit_step1.py --data-root data/step1 --vlguard-root data/step1/vlguard --output $report
if ($LASTEXITCODE -ne 0) { throw "只读数据审查失败" }
```

验收：VLGuard `validated` 且 `is_mock=false`；正式数量以新 manifest 为准，候选为 2,977 问答、1,999 图片。三组 plan 预计样本 34,585/42,106/2,977，steps 1,081/1,316/94；评测 smoke 18 请求；审查 `violations=[]`。HH 未解析分支仍作为限制记录，不强行清零。

## B. 本地：代码提交与推送

先查看清单，确认仅代码/配置/文档/测试。数据、环境、runs 和 key 被忽略，不进入 Git。提交范围由用户核对后执行。

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
git --no-pager diff --check
if ($LASTEXITCODE -ne 0) { throw "diff 检查失败" }
git --no-pager diff --stat
git status --short
# 核对清单后，再执行本阶段余下命令：
git add .gitignore README.md pyproject.toml requirements-train.txt configs docs scripts src tests
if ($LASTEXITCODE -ne 0) { throw "暂存失败" }
git --no-pager diff --cached --stat
git commit -m "Prepare Step 1 datasets, three LoRA recipes and evaluation workflow"
if ($LASTEXITCODE -ne 0) { throw "提交失败" }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "推送失败" }
git status
```

## C. 本地：离线数据打包与上传

A 阶段完成后才打包；包括独立 `vlguard` 子根。保留生成的实际包名，服务器阶段需要它。仅传输阶段，不接着运行训练。

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
$bundle = "step1-all-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".tar.gz"
$bundlePath = Join-Path (Get-Location).Path ("data\" + $bundle)
tar -czf $bundlePath -C data/step1 raw prepared vlguard
if ($LASTEXITCODE -ne 0) { throw "打包失败" }
$bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText($bundlePath + ".sha256", "$bundleHash  $bundle`n", [System.Text.UTF8Encoding]::new($false))
ssh tstar_001 "mkdir -p /data1/yaoziao/mm-sft-lab/data/transfer"
if ($LASTEXITCODE -ne 0) { throw "创建接收目录失败" }
scp $bundlePath ($bundlePath + ".sha256") "tstar_001:/data1/yaoziao/mm-sft-lab/data/transfer/"
if ($LASTEXITCODE -ne 0) { throw "上传失败" }
ssh tstar_001 "cd /data1/yaoziao/mm-sft-lab/data/transfer && sha256sum -c '$bundle.sha256'"
if ($LASTEXITCODE -ne 0) { throw "服务器校验失败" }
Write-Host "实际包名：$bundle"
```

## D. 服务器：同步与导入

先确保工作区干净，再 `git pull --ff-only`。下列包名必须替换为 C 阶段的实际包名；接收目录必须不存在，禁止覆盖旧 `data/imported`。

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  test -z "$(git status --porcelain)" || { echo "工作区不干净，请先检查"; exit 1; }
  git pull --ff-only origin main
  git log -1 --oneline
  BUNDLE='替换为实际step1-all包名.tar.gz'
  (cd data/transfer && sha256sum -c "$BUNDLE.sha256")
  test ! -e data/step1-imported-v1 || { echo "接收目录已存在，请使用新的导入目录"; exit 1; }
  mkdir data/step1-imported-v1
  tar -xzf "data/transfer/$BUNDLE" -C data/step1-imported-v1
  .venv-data/bin/python scripts/audit_step1.py --data-root data/step1-imported-v1 \
    --vlguard-root data/step1-imported-v1/vlguard --output runs/server-data-audit-v1.json
)
```

若该报告名已存在，选新报告名；导入后数据根为 `data/step1-imported-v1`，VLGuard 训练根为其 `/vlguard`。不要在服务器重新下载。

## E. 服务器：仅检查环境

先检测已有推理环境；不安装依赖、不加载训练权重、不更新参数。

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv/bin/python --version
  .venv/bin/python -m pip check
  .venv/bin/python -m pip show torch torchvision transformers accelerate peft safetensors Pillow
  nvidia-smi --query-gpu=index,name,memory.free,driver_version --format=csv -i 7
)
```

根据结果单独配置 `.venv-train`，保留已通过推理验证的 `.venv`。起始依赖见 `requirements-train.txt`；Torch/CUDA 不由该文件安装，目标为现有已验证的 torch 2.8.0+cu128 / torchvision 0.23.0 和 Python 3.11。服务器无数据源网络时需要匹配 Linux x86_64、CPython 3.11 的离线 wheel，不能复制 Windows venv。安装方案在此检测结果后确定，不把未经验证的环境创建命令混入 smoke。

后续示例都以已配置成功的 `.venv-train/bin/python` 为准。安装 PEFT 等依赖应在全量 B0 前完成，B0 与 adapter 复评使用同一冻结环境。

## F. 服务器：完整 token 审计

独立阶段，不更新模型。首先运行 `.venv-train/bin/python scripts/train.py preflight --config configs/train/textvqa_lora.json` 检查 CUDA/BF16/PEFT。随后三组分别 audit；示例是 TextVQA：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv-train/bin/python scripts/train.py audit --config configs/train/textvqa_lora.json \
    --data-root data/step1-imported-v1 --output runs/token-audit-textvqa-v1
)
```

HH 替换配置为 `hh_lora.json`、目录为 `token-audit-hh-v1`；VLGuard 替换配置为 `vlguard_lora.json`、数据根加 `/vlguard`、目录为 `token-audit-vlguard-v1`。只有 `passed`、`full_dataset=true`、`errors=0` 才可正式训练。超长样本先查看清单并明确处理协议；不自动缩图或删样本。代码、数据、配置或依赖变更后重新审计。

| 训练组 | config | 服务器 data-root | 全量 audit 输出 |
| --- | --- | --- | --- |
| T1 | configs/train/textvqa_lora.json | data/step1-imported-v1 | runs/token-audit-textvqa-v1 |
| H1 | configs/train/hh_lora.json | data/step1-imported-v1 | runs/token-audit-hh-v1 |
| V1 | configs/train/vlguard_lora.json | data/step1-imported-v1/vlguard | runs/token-audit-vlguard-v1 |

## G. 服务器：每组独立训练 smoke

环境和 token 审计通过后，只运行一个组的训练 smoke，检查结果再测试另两组。下面启动最多 64 条、2 个 updates：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv-train/bin/python scripts/train.py smoke --config configs/train/textvqa_lora.json \
    --data-root data/step1-imported-v1 --output runs/train-smoke-textvqa-v1 --limit 64 --steps 2
)
```

smoke 自行生成选中 64 条的审计，不传入全量 audit（它们选择集合不同）。检查语言 LoRA 参数范围、loss/梯度、冻结参数哈希、adapter 更新及 safetensors 一致性。HH/VLGuard 的配置和数据根按 F 阶段表述替换。smoke adapter 仅验收运行；恢复更新数值等价和重新加载推理另行验证，不能把它当正式结果。

## H. 服务器：冻结评测版本，先跑 B0

先使用 `configs/step1_smoke.json` 完成新的 18 请求推理验收，再以 `configs/step1_full.json` 做完整 B0。可在用户已有 tmux 会话中运行。生成、完整性复核、judge 选择/校准各自反馈，截断导致的 `needs_review` 不应被当作模型运行崩溃。

```bash
# 先在与 G 相同的环境变量配置下进入仓库，再单独运行该生成阶段。
.venv-train/bin/python scripts/lab.py generate --config configs/step1_full.json \
  --data-root data/step1-imported-v1 --run-dir runs/B0-step1-v1
```

生成完另行 `validate-run --run-dir runs/B0-step1-v1` 和 `score --run-dir runs/B0-step1-v1`。未 judge 的安全指标保持 pending。`export-review` 可导出人工模板；`judge` 需要先选定并配置兼容服务，不在本手册自动发送数据或调用服务。冻结同一 judge 协议后给 B0/T1/V1/H1 使用。

## I. 服务器：正式训练与 checkpoint 恢复

完整 B0 和训练 smoke 验收后，三组分别从同一原始模型开始；每组独立 audit/run。以下只启动 TextVQA：

```bash
.venv-train/bin/python scripts/train.py train --config configs/train/textvqa_lora.json \
  --data-root data/step1-imported-v1 --audit-dir runs/token-audit-textvqa-v1 --output runs/T1-textvqa-v1
```

HH 对应 H1，VLGuard 对应 V1；使用各自配置、audit、数据根和新 output，绝不将上一组 adapter 当起点。每 100 updates 和结束时保存 checkpoint。中断后从该 run 最后一个哈希校验完整的 checkpoint 加 `--resume-from`，配置、数据、依赖和代码须保持一致。保留已有日志；最终预算使用完成 manifest 的 progress，不累加重放 session 日志。

正式环境下先安排中断/恢复对照 smoke，验证恢复后的 updates 与连续运行在既定容差内一致，之后再依赖长训练恢复。当前代码仅有离线契约测试，不宣称该数值验证已完成。

## J. 服务器：adapter 重载与统一复评

在忽略的 `configs/local/` 复制 smoke/full 评测配置，只添加 `adapter_path` 指向正式 run 的 `adapter`。保持所有生成参数和父评测数据根一致；使用同一 `.venv-train`。先重载 smoke，再 full，最后同一 judge/score。TextVQA Test 使用 `configs/textvqa_test.json` 独立生成和 `export-textvqa-test` 导出，官方评分另行提交。

每个模型记录 adapter 哈希、数据 manifest、评测 config、代码/依赖、judge 身份和逐样本结果。最终表报告 SIUO 不安全率、MSS 派生危险服从率及官方指标、Wild harmful ASR/benign 拒答率和 TextVQA accuracy；同时保留截断、失败和判分 coverage。
