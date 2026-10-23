# Step 1 分阶段执行手册

> 文档状态（2026-10-23）：Step 1 / 旧 HF 操作记录。当前训练暂缓，baseline 请使用 [当前手册](baseline-runbook.md)，勿从下方旧阶段直接启动。

适用于本地 Windows `D:\Desktop\AI Safety\mm-sft-lab` 与服务器 `/data1/yaoziao/mm-sft-lab`，SSH 别名 `tstar_001`。每个阶段执行并反馈后再进入下一阶段，不能整篇一次粘贴运行。数据规则和实验限制见 [独立记录](data-and-experiment-notes.md)。

助手已完成本地代码与离线测试；以下真实准备、Git、传输、环境配置和模型运行由用户执行。当前旧服务器为 `fb33633`，需要先同步新代码。

2026-10-16 当前进度与顺序更新：服务器已同步 Step 1 和 processor 修复，数据导入审查、训练环境 preflight、processor 探针通过。用户要求优先完成原始模型 B0 baseline；现在执行 H 阶段的 18 条 smoke、全量生成和独立判分，TextVQA Test 按 J 阶段导出官方提交。F/G/I 的 token 预审、LoRA 排查和训练延后；无需为当前 B0 重装环境或运行训练入口。

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

2026-10-16 用户已完成 Conda `--clone --offline --copy` 创建 `.venv-train`，上传 PEFT 0.17.1 的通用 `py3-none-any.whl` 并以 `pip --no-index --no-deps` 单独安装；pip check 和 train preflight 通过。保留原环境的 Accelerate 1.15.0 / safetensors 0.8.0。真实训练尚未验证，具体状态见独立记录第 10 节。

后续示例都以已配置成功的 `.venv-train/bin/python` 为准。安装 PEFT 等依赖应在全量 B0 前完成，B0 与 adapter 复评使用同一冻结环境。

## F. 服务器：完整 token 审计

独立阶段，不更新模型。环境检查通过后先检查 processor，再做各组 64 条预审，最后才进入完整 audit。2026-10-16 首次预审因像素预算字段不同步停止；同步修复后独立运行下面的 CPU 探针，并查看结果：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  .venv-train/bin/python scripts/train.py processor-check --config configs/train/textvqa_lora.json
)
```

探针应返回 `passed`，两张合成图片的实际 `resized_pixels` 均在 65,536–1,048,576 内；它仅检查图像缩放，不证明完整 chat 编码或训练通过。再将下方 audit 命令增加 `--limit 64`，对三组各自使用新的输出目录预审。通过后去掉 limit 进行完整审计；示例是 TextVQA：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
# 2026-10-18：已完成模块并行判分

主生成仍在 `runs/B0-full-20261016T091009Z/main` 续跑。用户已通过视觉探针 `vision-judge-probe-20261018T041051.839029Z/report.json`，本轮只评分完整 MSSBench 与 SIUO。此处不用 git pull 或安装 SDK；用户上传操作脚本即可，避免修改运行中的推理包指纹。

本地 PowerShell：

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
scp ".\scripts\judge_partial.py" ".\scripts\judge_completed_server.sh" "tstar_001:/data1/yaoziao/mm-sft-lab/runs/recovery-tools/"
if ($LASTEXITCODE -ne 0) { throw "判分工具上传失败" }
```

服务器：

```bash
cd /data1/yaoziao/mm-sft-lab
bash runs/recovery-tools/judge_completed_server.sh
```

脚本在缺少环境密钥时隐藏输入；复用通过探针、冻结完整行快照、确认模块全部成功生成，前台检查 MSSBench/SIUO 各一条后 nohup 续判。有效 timeout=120、无自动重试，失败保留 journal。所选模块合计 2,127 个语义判断，SIUO MCQA 167 条本地计分；不判 WildJailbreak、不用 GPU、不修改生成锁/配置/目录。启动器控制锁防止重复启动；不要为了续判盲删锁。

查看本次判分：

```bash
JOB_ROOT=/data1/yaoziao/mm-sft-lab/runs/B0-full-20261016T091009Z
JUDGE_ROOT=$(cat "$JOB_ROOT/latest-completed-judge.txt")
tail -n 20 "$JUDGE_ROOT/judge.log"
cat "$JUDGE_ROOT/judge/module-metrics.json"
```

module-metrics.json 在前台两条检查后已生成，先显示不完整，后台结束时更新；以 requested_modules_complete 为准。metrics.json 显示全 baseline 覆盖率，因其他模块尚未完成而保持 complete=false 是正常的。

若后台任务停止，先检查 judge.log 和锁的所有者；同一快照用 `scripts/judge_partial.py judge --resume --retry-errors` 续判，保持原 --benchmarks mssbench siuo、--vision-probe-report、snapshot-dir 和 judge-dir。换新快照可显式 --reuse-judge-dir 复用同来源/服务/协议/回答哈希的成功判分。本操作工具和普通 lab.py judge 的签名不同，不直接混放其 judgments。

## 2026-10-18 内容审核中断后的版本迁移恢复

用户回传原 completed-modules-20261018T051824-1172696 中只有 7 成功、1 个 data_inspection_failed；不能直接用新脚本 --resume 该旧目录，因为原签名绑定旧脚本 SHA256。以下操作保留旧文件，显式校验后在新目录导入成功与审核拒绝，优先 MSSBench，跳过已知拒绝项并继续其他 SIUO。

本地 PowerShell：

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
scp ".\scripts\judge_partial.py" "tstar_001:/data1/yaoziao/mm-sft-lab/runs/recovery-tools/judge_partial_v2.py"
if ($LASTEXITCODE -ne 0) { throw "新版判分工具上传失败" }
scp ".\scripts\recover_completed_judge_server.sh" "tstar_001:/data1/yaoziao/mm-sft-lab/runs/recovery-tools/recover_completed_judge_server.sh"
if ($LASTEXITCODE -ne 0) { throw "恢复启动器上传失败" }
```

服务器：

```bash
cd /data1/yaoziao/mm-sft-lab
bash runs/recovery-tools/recover_completed_judge_server.sh
```

启动器离线 plan 校验不发送 API，按当前回传应显示 reuse_successful_requests=7、reuse_blocked_requests=1、remaining_requests_to_attempt=2119。通过后 nohup 运行；当前 judge 指针更新到新的 judge-recovery-* 目录。主生成不变，不使用 GPU，不做 git pull 或依赖安装。

```bash
JOB_ROOT=/data1/yaoziao/mm-sft-lab/runs/B0-full-20261016T091009Z
JUDGE_ROOT=$(cat "$JOB_ROOT/latest-completed-judge.txt")
tail -n 20 "$JUDGE_ROOT/judge.log"
```

结束后查看 judge/module-metrics.json 和 judge/blocked-samples.jsonl。新任务途中停止时，先看错误与锁状态；同一 v2 目录应使用 v2 工具 --resume --retry-errors，保留 snapshot-dir、judge-dir、模块和视觉报告参数，不再传 --reuse-legacy-judge-dir，不重新运行迁移启动器造成重复计费。已知 provider_content_filter 即使显式 retry-errors 也默认不再发送。

## 2026-10-19：文件超限后的 v3 恢复

用户回传 MSSBench 已完整、成功总数 1,990，内容审核拒绝 2、文件超限 1、未尝试 134。v2 在 siuo/gen/2014 文件超限后停止。新版精确分类输入大小限制，保留原输入，导入已完成标签，不重判 MSSBench。

本地 PowerShell（保留服务器 v1/v2 文件）：

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"
scp ".\scripts\judge_partial.py" "tstar_001:/data1/yaoziao/mm-sft-lab/runs/recovery-tools/judge_partial_v3.py"
if ($LASTEXITCODE -ne 0) { throw "v3 判分工具上传失败" }
scp ".\scripts\recover_input_limit_judge_server.sh" "tstar_001:/data1/yaoziao/mm-sft-lab/runs/recovery-tools/recover_input_limit_judge_server.sh"
if ($LASTEXITCODE -ne 0) { throw "文件超限恢复启动器上传失败" }
```

服务器：

```bash
cd /data1/yaoziao/mm-sft-lab
bash runs/recovery-tools/recover_input_limit_judge_server.sh
```

先 plan-only 校验 v2 脚本/签名基础、同一服务与来源回答，按当前结果应显示 reuse_successful_requests=1990、reuse_blocked_requests=2、reuse_input_limit_requests=1、remaining_requests_to_attempt=134。通过后在新 judge-input-recovery-* 目录后台运行。原 snapshot 与视觉探针从旧 v2 manifest 读取，旧文件不修改。

继续使用 latest-completed-judge.txt 定位新任务，查看 judge.log。完成后 manifest 的 processing_complete=true 表示所有目标已判或明确平台缺失；如果仍有文件超限，status=finished_with_provider_errors，scoring_complete=false 是预期。

输出新增 provider_input_limit_requests、input-limit-samples.jsonl 与 provider-unscored-samples.jsonl；blocked-samples.jsonl 仍只统计内容审核。旧检查脚本若只把成功数与审核拦截数相加，会漏掉文件超限，不能继续用该旧公式判断处理是否结束。请同时检查三项，或直接查看新版 manifest/module-metrics.json 的 processing_complete。

## 2026-10-20：401 后原 v3 目录续判

最新错误为 siuo/gen/2015 的 invalid_api_key。已有 1,990 成功、2 审核、1 超限、1 认证错误、133 未尝试；更新有效 Key 后共 134 条待尝试。无需上传新脚本、修改签名或重建任务。以下在服务器输入密钥，仅进入进程环境；前台只重试一条，失败时不启动后台任务，成功记录保留并跳过。

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  umask 077
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME DASHSCOPE_API_KEY
  export PYTHONNOUSERSITE=1

  JOB_ROOT="$PWD/runs/B0-full-20261016T091009Z"
  WORK=$(cat "$JOB_ROOT/latest-completed-judge.txt")
  PY="$PWD/.venv-data/bin/python"
  TOOL="$PWD/runs/recovery-tools/judge_partial_v3.py"
  test -f "$WORK/judge/manifest.json"
  test -f "$TOOL"

  exec 9>"$JOB_ROOT/.completed-judge.control.lock"
  if ! flock -n 9; then
    echo "已有判分任务运行，不重复启动。"
    exit 1
  fi
  if [ -e "$WORK/judge/.judge.lock" ]; then
    echo "判分目录仍有锁，请先核对其进程，不删除锁。"
    exit 1
  fi

  read -r -s -p "请输入有效的 DashScope API Key（输入不可见）: " DASHSCOPE_API_KEY
  printf '\n'
  test -n "$DASHSCOPE_API_KEY" || { echo "Key 不能为空"; exit 1; }
  export DASHSCOPE_API_KEY
  export JUDGE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
  export JUDGE_MODEL="deepseek-v4.1-flash"

  manifest_field() {
    "$PY" - "$WORK/judge/manifest.json" "$1" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())[sys.argv[2]])
PY
  }
  SNAPSHOT=$(manifest_field snapshot_dir)
  VISION_REPORT=$(manifest_field vision_probe_report)
  ARGS=(judge --snapshot-dir "$SNAPSHOT" --judge-dir "$WORK/judge"
    --benchmarks mssbench siuo --vision-probe-report "$VISION_REPORT"
    --resume --retry-errors)

  CHECK_LOG="$WORK/key-check-$(date -u +%Y%m%dT%H%M%S).log"
  if ! "$PY" -u "$TOOL" "${ARGS[@]}" --limit 1 > "$CHECK_LOG" 2>&1; then
    head -n 1 "$CHECK_LOG"
    tail -n 1 "$WORK/judge/judgments.jsonl"
    echo "单条检查未通过，未启动后台续判。日志：$CHECK_LOG"
    exit 1
  fi
  head -n 1 "$CHECK_LOG"

  nohup "$PY" -u "$TOOL" "${ARGS[@]}" >> "$WORK/judge.log" 2>&1 < /dev/null &
  JUDGE_PID=$!
  printf '%s\n' "$JUDGE_PID" > "$WORK/judge.pid"
  sleep 2
  echo "续判目录：$WORK"
  echo "判分 PID：$JUDGE_PID"
  tail -n 5 "$WORK/judge.log"
)
```

这会重试认证失败及其他普通错误，跳过成功记录和已知 provider_content_filter/provider_input_limit。Key 仍无效时保留失败记录并停止；不输出 Key，不将 401 赋安全标签。不要把实际 Key 发到聊天。

## 2026-10-20：仅重试两条 judge 解析失败

最新状态已变为 2,122 条成功、2 条内容审核、1 条文件超限、2 条普通解析错误，未尝试为 0。SIUO gen 当前为 162/167，有效子集不安全 87/162（53.70%）。旧认证错误已恢复；不要再运行上一节“单条检查后后台全量”的启动块。

以下直接在服务器前台运行，最多两次 API 请求，保持 v3 版本/签名、原目录和原评测输入。Key 重新隐藏输入，避免上次子 shell 的导出未作用到当前终端。若计数已变化或锁仍存在则退出，不删锁、不重迁移。脚本和依赖均不修改。

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  umask 077
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME DASHSCOPE_API_KEY
  export PYTHONNOUSERSITE=1
  export JUDGE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
  export JUDGE_MODEL="deepseek-v4.1-flash"

  exec 9>runs/B0-full-20261016T091009Z/.completed-judge.control.lock
  flock -n 9 || { echo "已有判分任务运行，不重复启动"; exit 1; }
  read -r -s -p "请输入 DashScope API Key（输入不可见）: " DASHSCOPE_API_KEY
  printf '\n'
  test -n "$DASHSCOPE_API_KEY" || { echo "Key 不能为空"; exit 1; }
  export DASHSCOPE_API_KEY

  .venv-data/bin/python - <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path("runs/B0-full-20261016T091009Z").resolve()
work = Path((root / "latest-completed-judge.txt").read_text().strip())
judge = work / "judge"
manifest = json.loads((judge / "manifest.json").read_text())
if (judge / ".judge.lock").exists():
    raise SystemExit("判分锁仍存在，请先检查进程，不删除锁")
if manifest.get("actionable_error_requests") != 2 or manifest.get("unattempted_requests") != 0:
    raise SystemExit("当前计数已变化，请重新检查，未发送 API 请求")

log = work / ("format-retry-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".log")
args = [sys.executable, "-u", "runs/recovery-tools/judge_partial_v3.py", "judge",
        "--snapshot-dir", manifest["snapshot_dir"], "--judge-dir", str(judge),
        "--benchmarks", "mssbench", "siuo",
        "--vision-probe-report", manifest["vision_probe_report"],
        "--resume", "--retry-errors", "--limit", "2"]
print("仅重试两条解析失败，结果保存在：", log, flush=True)
with log.open("x", encoding="utf-8") as stream:
    result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
for line in log.read_text(encoding="utf-8").splitlines():
    if line.startswith("judge ["):
        print(line)

manifest = json.loads((judge / "manifest.json").read_text())
report = json.loads((judge / "module-metrics.json").read_text())
fields = ("status", "successful_requests", "expected_requests", "provider_blocked_requests",
          "provider_input_limit_requests", "actionable_error_requests", "unattempted_requests",
          "processing_complete", "scoring_complete", "last_session_attempted")
print("退出码：", result.returncode)
print(json.dumps({k: manifest.get(k) for k in fields}, ensure_ascii=False, indent=2))
print("报告签名一致：", report["judge_signature"] == manifest["judge_signature"])
print("SIUO gen：")
print(json.dumps(report["tasks"]["siuo/gen"], ensure_ascii=False, indent=2))
print("日志：", log)
PY
)
```

两条都成功时预期为 2,124/2,127、SIUO 164/167，processing_complete=true、scoring_complete=false；剩余三条是平台缺失，仍须单列，不能据此称 SIUO 全量完成。若仍有解析失败，先检查日志/journal 中该两条的 raw_judgment、judge_finish_reason 与错误，不自动改标签或无限重试。

## 2026-10-20：重试完成后只读取得 SIUO 分数

用户回传已经达到 2,124/2,127、processing_complete=true、普通错误 0、未尝试 0，judge 锁不存在且无运行进程。本轮处理已结束，不再执行前面的重试块。SIUO gen 当前为 164/167（98.2036%），新分数需读取更新后的报告；此前 87/162（53.70%）仅是补判前快照。

以下只读取文件，不调用 API，无需输入 Key：

```bash
cd /data1/yaoziao/mm-sft-lab
.venv-data/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("runs/B0-full-20261016T091009Z")
work = Path((root / "latest-completed-judge.txt").read_text().strip())
judge = work / "judge"
manifest = json.loads((judge / "manifest.json").read_text())
report = json.loads((judge / "module-metrics.json").read_text())
if report["judge_signature"] != manifest["judge_signature"]:
    raise SystemExit("报告与 manifest 签名不一致，暂不使用此报告")
print("判分目录：", work)
print("报告签名一致：True")
print("=== SIUO 补判后指标 ===")
print(json.dumps(report["tasks"]["siuo/gen"], ensure_ascii=False, indent=2))
PY
```

将此输出用于更新不安全率、有效率、联合指标及全量分母下的缺失取值范围；三个平台缺失保持原清单。其他完成模块的分数不因仅重试两条 SIUO 而改变。

结果回传完成：报告与 manifest 签名一致，SIUO 不安全 89/164（54.2683%），覆盖率 164/167（98.2036%），有效率 100/164（60.9756%），安全且有效 74/164（45.1220%）。全量缺失取值范围为 89/167–92/167（53.2934%–55.0898%），不是置信区间。当前所选模块处理已结束，上述读取命令留作审计复查；本轮不再执行重试。详细汇报见 [B0 结果](b0-progress-2026-10-19.md)。
