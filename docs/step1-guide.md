# Step 1：本地数据、离线检查与后续训练

更新：2026-10-15。当前入口覆盖 B0 原始模型、T1 TextVQA、V1 VLGuard、H1 HH-harmless-chosen。三个训练组各自使用完整可用训练划分训练 1 epoch，从同一原始模型独立初始化。数据处理与特殊约定见 [独立记录](data-and-experiment-notes.md)，最新分阶段命令见 [服务器执行手册](server-runbook.md)。

四个新来源已下载，TextVQA/HH/Wild 已完成真实准备；VLGuard 转换入口已补齐，真实准备由用户执行。真实 LoRA 训练、GPU 恢复和 adapter 重载推理尚未验证。合成测试不能产生真实模型分数。

每个编号是一阶段，执行结束、反馈结果后再进行下一阶段。助手本地改代码；用户运行下载、Git、SCP 与服务器命令。

## 1. 本地离线验收

本机已知可用解释器是 `C:\anaconda3\python.exe`；原 `.venv-data` 启动器指向缺失的 Python，不用于下面的本地命令。离线检查只需要 Python 和 Pillow，不需要 torch、权重、网络或 API key。

本地 PowerShell：

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
$python = "C:\anaconda3\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "找不到已知 Python，请先确认解释器路径" }
$checkDir = "runs/local-step1-check-" + (Get-Date -Format "yyyyMMdd-HHmmss")
& $python scripts/check_step1.py --output $checkDir
if ($LASTEXITCODE -ne 0) { throw "离线验收失败，请保留并贴出输出" }
```

输出包括完整单元测试、三组合成数据准备与只读审查、mock 推理/mock judge、TextVQA Val 本地评分、人工标签模板及三个训练 plan。摘要必须有 `is_mock: true`、`real_training_verified: false`。不会安装依赖、下载真实数据或启动训练。

## 2. 本地下载环境

本地 PowerShell，仅在上一阶段通过后执行：

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
if (-not (Test-Path -LiteralPath ".venv-step1")) {
    & "C:\anaconda3\python.exe" -m venv .venv-step1
    if ($LASTEXITCODE -ne 0) { throw "创建本地数据环境失败" }
}
$python = ".\.venv-step1\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "环境目录存在但解释器缺失，请检查后再处理" }
& $python -m pip install -r requirements-data.txt
if ($LASTEXITCODE -ne 0) { throw "安装数据依赖失败" }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "依赖检查失败" }
Get-PSDrive -Name D
```

TextVQA 图片包、解压图片和离线归档会同时占用空间，请为数据及临时文件预留几十 GB，并以下载后的实际大小为准。数据环境忽略于 Git。

## 3. 本地逐源下载

用户已选择将四个下载放在同一个阶段顺序执行：TextVQA → HH harmless → WildJailbreak eval → VLGuard train。某个来源发生网络/权限等错误时记录失败并继续，最后汇总；不自动接着解压、准备或训练。下载根目录统一为 `data/step1`，不会覆盖已有 `data/prepared` 或服务器 `data/imported`。下面的单源命令也保留用于重试。

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
& ".\.venv-step1\Scripts\python.exe" scripts/download_step1.py --all --data-root data/step1
$downloadExitCode = $LASTEXITCODE
if ($downloadExitCode -eq 0) {
    Write-Host "四个数据集全部下载并校验完成"
} else {
    Write-Warning "下载未全部完成，请查看汇总并贴出输出；已完成的数据保留"
}
```

逐源结果实时写入 `data/step1/download_batch_summary.json`；全部尝试结束后才返回整体失败码。失败项可以用 `--dataset` 单独重试，或重跑 `--all`（会校验并复用已完成来源）。Ctrl+C 是用户主动停止，会终止整批，保留已经完成的文件。WildJailbreak 未完成访问授权时会单独失败，其他来源下载不受该权限错误影响；访问授权方法见 3c。

### 3a. TextVQA

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
& ".\.venv-step1\Scripts\python.exe" scripts/download_step1.py --dataset textvqa --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "TextVQA 下载失败" }
```

下载固定 v0.5.1 train/val/test 标注和 train_val/test 图片 ZIP，记录逐文件 SHA256。CDN 下载使用 `.part` 文件，在服务器确认 Range offset 时续传；已完成文件复用前重新核对哈希。首次下载用内容哈希冻结静态 URL 资产，不把静态 URL 称为 Git commit。[官方地址依据](https://huggingface.co/datasets/facebook/textvqa/blob/main/textvqa.py)

### 3b. HH-RLHF harmless

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
& ".\.venv-step1\Scripts\python.exe" scripts/download_step1.py --dataset hh --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "HH harmless 下载失败" }
```

只下载 `harmless-base/train.jsonl.gz`、`test.jsonl.gz` 和 README；test 仅用于上下文去污染，不加入训练或主评测表。不下载 helpful、red-team 或其他数据。[官方文件](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/harmless-base)

### 3c. WildJailbreak eval

先由用户登录 [官方数据页](https://huggingface.co/datasets/allenai/wildjailbreak)，接受其数据访问条件。它与 VLGuard 的授权是两回事。然后在本机交互式登录，token 不发到聊天或配置文件：

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
& ".\.venv-step1\Scripts\hf.exe" auth login
if ($LASTEXITCODE -ne 0) { throw "Hugging Face 登录失败" }
```

授权完成后单独下载：

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
& ".\.venv-step1\Scripts\python.exe" scripts/download_step1.py --dataset wildjailbreak --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "WildJailbreak eval 下载失败，请核对账号访问权限" }
```

仅下载 `eval/eval.tsv` 和 README，固定 HF commit；禁止把 WildJailbreak train 用作此轮评测。HF 每源独立保存 source lock，失败后复用原 commit，不自动更新到新的 main。

### 3d. VLGuard train

用户已在本机登录 Hugging Face，并成功下载小型 `train.json` 完成访问检查。正式下载使用同一登录凭据，将文件单独保存到 `data/step1/raw/vlguard`；访问检查目录 `data/access-check/vlguard` 不作为完整数据快照。

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
& ".\.venv-step1\Scripts\python.exe" scripts/download_step1.py --dataset vlguard --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "VLGuard 下载失败，请保留输出并核对账号访问权限" }
```

仅选择 `train.json`、训练图片 `train.zip` 和 `README.md`，锁定 HF commit 并记录文件 SHA256；训练标注与图片都齐全才记为成功。本阶段不下载 test 或 LLaVA 混合数据。[官方文件与访问条件](https://huggingface.co/datasets/ys-zong/VLGuard/tree/main)

VLGuard 已提供独立 `scripts/prepare_vlguard.py` 和 `configs/train/vlguard_lora.json`。父准备入口继续转换 TextVQA/HH/Wild，VLGuard 写入 `<父根>/vlguard`，不改父 prepared。真实准备命令见执行手册 A 阶段。

## 4. 准备完整数据快照

TextVQA、HH 和 WildJailbreak 下载完整后，可准备父数据快照；此入口不转换或要求 VLGuard。`--benchmark-data-root data` 指向现有已验证的 SIUO/MSSBench 数据，准备程序将旧图像以硬链接（不可用时复制）纳入新的可移植根目录，并重新核对哈希。父快照已由用户准备并通过验证，不需要为新增 VLGuard 重跑此阶段。

真实准备阶段会输出 `[prepare]` 进度：逐源哈希校验、图片解压、评测/训练转换及快照写入；图片校验每 2,000 张输出一次。TextVQA 图片包较大，源文件哈希及解压阶段需要等待。

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
& ".\.venv-step1\Scripts\python.exe" scripts/prepare_step1.py --data-root data/step1 --benchmark-data-root data
if ($LASTEXITCODE -ne 0) { throw "Step 1 数据准备失败" }

& ".\.venv-step1\Scripts\python.exe" scripts/lab.py plan --config configs/step1_smoke.json --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "评测数据 plan 失败" }

& ".\.venv-step1\Scripts\python.exe" scripts/train.py plan --config configs/train/textvqa_lora.json --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "TextVQA 训练数据 plan 失败" }
& ".\.venv-step1\Scripts\python.exe" scripts/train.py plan --config configs/train/hh_lora.json --data-root data/step1
if ($LASTEXITCODE -ne 0) { throw "HH 训练数据 plan 失败" }
```

输出在 `data/step1/prepared`：

- `train_textvqa.jsonl` / `train_hh.jsonl`：训练记录；TextVQA 在原始非空白标注中确定性选 normalized 最高频答案，频次/字典序处理平票并保留原始表述。`a`、`the`、`a+` 等评分规范化后为空的有效 OCR 答案仍可作为非空训练目标；不将评分规范化结果直接作为目标文字。
- `inputs.jsonl` / `evaluation.jsonl`：仅模型输入与独立评测标签；TextVQA Test 的答案为 null，不伪造分数。
- `excluded_training.jsonl` / `decontamination.json`：过滤样本及原因、原始/可用训练规模；HH 格式异常附原始 source index、一基行号及错误详情。
- `hh_test_context_issues.jsonl`：HH test 无法解析的上下文，按 chosen/rejected 分支单列；test 不用于训练。
- `manifest.json`：数据源、逐文件/图像哈希、转换规则、审计和输出哈希。

过滤包含 TextVQA heldout image ID、所有评测图片的精确哈希、HH test 两支可解析上下文、HH 最后用户问题与评测 query 的规范化精确重复，以及训练集内部重复。近似图片/改写语义去污染尚未实现，审计中明确记录。HH chosen/rejected 必须具有相同历史；准备阶段排除并记录非交替角色、空轮次、控制 token 或两支历史不一致的训练行，不拼接修补错误历史。默认底层解析接口仍严格报错。

2026-10-15 本机真实 HH 扫描发现 train 共 42,537 行，其中 279 行不符合严格对话格式、2 行历史不一致；最终训练量还须扣除去重与去污染。test 最终回答为空时仍保留可解析上下文；无法解析的分支单独记录，其上下文未纳入该项精确去污染检查，不能宣称已完整覆盖所有 test 上下文。原始/格式合格/排除/最终训练数量均进入审计。

重复准备相同数据会保留 manifest；变更数据/转换结果时需使用新的 data root，禁止无声覆盖。`--allow-unlocked` 只供合成测试；即使关闭配置的 locked 检查，真实训练入口仍拒绝 mock 数据。

正式数据未过滤前，`step1_full.json` 总请求 9,504（旧 2,294 + Wild 2,210 + TextVQA Val 5,000）。`step1_smoke.json` 按 limit 2 选 18 请求；MSSBench 保留完整配对。TextVQA Test 5,734 请求使用单独 `textvqa_test.json`，不使本地 Val 报表一直处于等待官方 Test 评分的状态。

## 5. Git 同步与离线数据传输

完成上面阶段后再单独执行 Git 提交、推送、服务器拉取；助手不代执行。先 `git --no-pager diff --stat` 和 `git status --short` 确认修改清单，确保没有数据、环境、key 或 run 被纳入。

先完成执行手册 A 阶段的 VLGuard 子快照，再打包；需包括 raw、prepared 和 vlguard 三个目录。数据打包示例（本地 PowerShell，单独传输阶段）：

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
$bundle = "step1-offline-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".tar.gz"
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
if ($LASTEXITCODE -ne 0) { throw "服务器哈希校验失败" }
```

服务器接收后再给出带实际包名的解压指令，使用新的空目录，例如 `data/step1-imported`。不要覆盖旧 `data/imported`。运行 inference/train 的 `--data-root` 指向新根目录，prepared 中路径可移植。

## 6. 服务器环境检测：不启动训练

先准备独立的 `.venv-train`。`requirements-train.txt` 是待服务器验证的固定依赖起点；CUDA torch 2.8.0+cu128 / torchvision 0.23.0 需要另行安装/复用。离线 wheel 必须匹配 Linux、架构和 Python 3.11，不复制 Windows venv。保留已经通过推理验证的 `.venv`；具体离线安装命令在环境检测结果确认后提供。

以下服务器 Bash 命令假设 `.venv-train` 已完成依赖安装，仅检测 CUDA、BF16、文件和 PEFT 导入，不加载训练模型、计算 loss 或更新权重：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv-train/bin/python -m pip check
  .venv-train/bin/python scripts/train.py preflight --config configs/train/textvqa_lora.json
)
```

## 7. 训练 token 审计：不更新模型

环境通过后，每个数据组独立执行一次全量 CPU processor 审计。下面以 TextVQA 为例，HH 替换配置和审计目录；VLGuard 替换配置并将数据根改为父根下 `/vlguard`。`--data-root` 使用服务器实际解压目录。

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv-train/bin/python scripts/train.py audit \
    --config configs/train/textvqa_lora.json --data-root data/step1-imported \
    --output runs/token-audit-textvqa-v1
)
```

检查完整序列长度、模板前缀一致、答案监督 token 非零、图像输入存在。总序列上限 4096，含视觉 token、问题、历史、答案和模板；没有自动截断或丢弃，超长/格式错误写到 `errors.jsonl`，返回失败。先看清单再讨论处理策略。正式训练要求 full_dataset=true 且全部通过的 audit；代码、数据、模型、配置与依赖须一致，CPU audit 的 GPU 可见性可以不同。

## 8. 单独训练 smoke

只用训练划分最多 64 条，最多 2 个 optimizer updates；不会把 smoke adapter 当作正式结果。三个组分别运行并检查，下面只启动 TextVQA：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv-train/bin/python scripts/train.py smoke \
    --config configs/train/textvqa_lora.json --data-root data/step1-imported \
    --output runs/train-smoke-textvqa-v1 --limit 64 --steps 2
)
```

验收包括：trainable_parameters 只包含语言层 q/k/v/o LoRA；全部冻结参数训练前后哈希相同；loss/梯度有限且非零；adapter 更新；磁盘 safetensors 与内存 adapter 完全一致。视觉模块同时保持 eval 模式。这里只检查序列化一致性，重新加载后的真实推理要用单独 inference smoke 验收。

## 9. 正式训练与恢复：之后执行

完整 baseline 与训练 smoke 验收后再启动正式训练。配置遵循用户指定 LoRA 16/32/0.05、LR 5e-5、AdamW、1 epoch、有效 batch 32、4096。

新增显式起始值：micro-batch 1 × 累积 32，语言 attention q/k/v/o projection，bias none，seed 42，linear scheduler，warmup ratio 0.03（步数向下取整），weight decay 0，Adam betas 0.9/0.999、epsilon 1e-8、clip 1，训练像素面积 65,536–1,048,576。图像预算是此轮建议值，首次 GPU 测试前核对；不沿用 16M 像素的推理默认上限来挤占 4096 训练序列。

训练 loss 只覆盖最后 assistant 回答及其终止 token。每个 micro loss 先转成监督 token 的总 loss，累积后按该 optimizer window 的实际监督 token 总数归一化；尾部不足 32 样本的 window 保留，步数为 ceil(N/32)。三个全量 1 epoch 组预算不同，记录实际 token/更新步数，不能称为预算对齐。

服务器 Bash（TextVQA，未来阶段）：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv-train/bin/python scripts/train.py train \
    --config configs/train/textvqa_lora.json --data-root data/step1-imported \
    --audit-dir runs/token-audit-textvqa-v1 --output runs/train-textvqa-v1
)
```

HH 使用 `configs/train/hh_lora.json`，VLGuard 使用 `configs/train/vlguard_lora.json` 且数据根为 `<父根>/vlguard`；各自独立 audit/run，从原始模型重新初始化。OOM 会明确失败，代码不会偷偷调小分辨率或改变有效 batch。

每 100 updates 和结束时保存 `checkpoint-XXXXXX`，含 adapter、processor、optimizer、scheduler、已处理样本偏移和 RNG。中断后以同一配置/数据/环境/代码和 output 加 `--resume-from runs/train-textvqa-v1/checkpoint-实际步数`；不能把另一组 adapter 当作续训 checkpoint。

`updates.jsonl` 保留 session ID；从旧 checkpoint 恢复时可能重新执行之后未保存的 updates，原日志不删除，汇总使用最后完成的 manifest/progress，不能简单相加所有 session 的 token 数。checkpoint 缺失、未完成或哈希不符会拒绝恢复。

## 10. 评测 adapter、人工 judge 与 Test 导出

复制评测 JSON 到忽略的 `configs/local/`，添加 `adapter_path` 指向训练 run 的 `adapter`，其余生成设置保持不变。加载时核对训练 base 指纹并哈希 adapter 权重。先用 `step1_smoke.json` 做真实重载推理，再运行 `step1_full.json`。

所有对照在同一冻结评测代码与依赖下运行；使用 base+adapter 时，B0 之前就把 PEFT 加入该评测环境，避免训练后才安装依赖造成 environment 不一致。历史 48 条检查保留，不与新全量版本直接混表。可选 `scripts/train.py merge --config ... --adapter-dir ... --output 新空目录` 导出完整权重；BF16 合并需另做推理验证。

无需 API 的人工判分：

```bash
python scripts/lab.py export-review --run-dir runs/实际推理run --output runs/review-v1.jsonl
# 人工只填写每行 labels；保留题目、图片、参考信息、回答和哈希。
python scripts/lab.py import-judgments --run-dir runs/实际推理run \
  --input runs/review-v1.jsonl --judge-dir runs/实际推理run/manual-v1 --reviewer reviewer1
python scripts/lab.py score --run-dir runs/实际推理run --judge-dir runs/实际推理run/manual-v1
```

SIUO 标签：`{"safety": true, "effectiveness": "very_effective"}`，effectiveness 也可为 partially_effective/ineffective。MSSBench：`{"class_index": 0}`，取值 0–3。WildJailbreak：`{"harmful": false, "refusal": true}`。`labels:null` 保留 pending，不当成安全答案。新一轮人工标签使用新 judge_dir。

API judge 仍可选：SIUO 需要图片支持；MSSBench 当前仅回答文本分类；Wild 发送 query+response，用本项目 adapted rubric 判 harmful/refusal。它不是官方 judge prompt 的逐字复现，正式使用前需人工校准。TextVQA 与 MCQA 不调用语义 judge。

TextVQA Test 单独生成后：

```bash
python scripts/lab.py export-textvqa-test --run-dir runs/实际TextVQATestRun --output runs/textvqa-test-submission.json
```

文件包含 question_id/answer；官方分数仍为 pending。先核对覆盖完整和平台当前提交入口，再由用户提交；不能把 mock 预测上传或把 Val 结果标成 Test。
