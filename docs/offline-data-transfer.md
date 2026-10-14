# 本地准备数据并传到服务器

服务器在解析 Hugging Face 数据源时发生连接超时。当前本地 Windows 下载、离线包上传、SHA-256 校验及服务器数据复核均已完成，见 [数据快照](data-snapshot.md)。以下保留可重复使用的操作流程；代码继续通过 Git 同步，`data/` 保持忽略。

## 1. 本地 Windows 下载

本地 PowerShell，在仓库根目录执行。使用独立 `.venv-data`，不改变 Anaconda base 中的依赖。

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"

if (-not (Test-Path -LiteralPath ".\.venv-data\Scripts\python.exe")) {
    if (Test-Path -LiteralPath ".\.venv-data") {
        throw ".venv-data 已存在但不是可用的 Windows Python 环境，请先检查"
    }
    python -m venv .venv-data
    if ($LASTEXITCODE -ne 0) { throw "创建数据环境失败" }
}

& ".\.venv-data\Scripts\python.exe" -m pip install -r requirements-data.txt
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

& ".\.venv-data\Scripts\python.exe" scripts/download_datasets.py --data-root data --siuo-revision 18974b65d238ad636d65d238541c7d75279ebb3e
if ($LASTEXITCODE -ne 0) { throw "数据下载失败，请保留完整输出" }

& ".\.venv-data\Scripts\python.exe" scripts/prepare_datasets.py --data-root data
if ($LASTEXITCODE -ne 0) { throw "数据校验失败" }
```

SIUO 标注固定到服务器已解析出的 SHA；两个 HF 来源由本地解析并写入 source lock。已有 lock 时仍按既定规则复用或拒绝冲突。成功需看到 `Download complete`、`validated` 和 `locked`。

## 2. 本地打包与 SHA-256

下载与准备成功后再执行。包中同时携带 prepared、原始数据和来源记录，使服务器无需联网重建数据。

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"

if (Test-Path -LiteralPath ".\data\benchmarks-offline.tar.gz") {
    throw "离线包已存在，请先确认是否复用，或为新包选择不同名称"
}

tar -czf "data/benchmarks-offline.tar.gz" -C "data" raw prepared sources.lock.json download_manifest.json
if ($LASTEXITCODE -ne 0) { throw "打包失败" }

$bundleHash = (Get-FileHash -LiteralPath ".\data\benchmarks-offline.tar.gz" -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumLine = "$bundleHash  benchmarks-offline.tar.gz`n"
$checksumPath = Join-Path (Get-Location).Path "data\benchmarks-offline.tar.gz.sha256"
[System.IO.File]::WriteAllText($checksumPath, $checksumLine, [System.Text.UTF8Encoding]::new($false))
Get-Item -LiteralPath ".\data\benchmarks-offline.tar.gz" | Select-Object Name, Length
Get-Content -LiteralPath ".\data\benchmarks-offline.tar.gz.sha256"
```

传输两个文件：`benchmarks-offline.tar.gz` 与 `benchmarks-offline.tar.gz.sha256`。本地 SSH 别名 `tstar_001` 已验证，远端用户为 `yaoziao`，目标主机为 `user-SYS-420GP-TNAR`。SCP 沿用此别名的 SSH 配置。

本地 PowerShell，在仓库根目录执行：

```powershell
ssh tstar_001 "mkdir -p /data1/yaoziao/mm-sft-lab/data/transfer"
if ($LASTEXITCODE -ne 0) { throw "创建服务器接收目录失败" }

scp ".\data\benchmarks-offline.tar.gz" ".\data\benchmarks-offline.tar.gz.sha256" "tstar_001:/data1/yaoziao/mm-sft-lab/data/transfer/"
if ($LASTEXITCODE -ne 0) { throw "上传失败，请保留输出" }

ssh tstar_001 "cd /data1/yaoziao/mm-sft-lab/data/transfer && sha256sum -c benchmarks-offline.tar.gz.sha256"
if ($LASTEXITCODE -ne 0) { throw "服务器 SHA-256 校验失败" }
```

## 3. 服务器接收与离线校验

先按上一步将两个文件放入 `/data1/yaoziao/mm-sft-lab/data/transfer/`。

服务器 Bash，校验包后解压到新目录，保留之前失败下载的目录：

```bash
(
  set -e
  cd /data1/yaoziao/mm-sft-lab/data/transfer
  sha256sum -c benchmarks-offline.tar.gz.sha256
  if [ -e ../imported ]; then
    echo 'data/imported 已存在，请先检查或选择新的导入目录'
    exit 1
  fi
  mkdir ../imported
  tar -xzf benchmarks-offline.tar.gz -C ../imported
  cd /data1/yaoziao/mm-sft-lab
  .venv-data/bin/python scripts/prepare_datasets.py --data-root data/imported --output-root data/imported/server-check
)
```

此复核使用服务器已有的数据准备脚本，重新检查图片可读性、标注与配对。输出写入独立的 `server-check/`，保留本地传来的 `prepared/` 快照，方便之后在本地 API judge。

同步包含 `scripts/lab.py` 和 `configs/baseline_smoke.json` 的框架代码后，再执行 `plan --config configs/baseline_smoke.json --data-root data/imported`，核对 prepared 文件与全部被引用图片的 SHA-256 和完整配对。

之后所有评测命令显式使用 `--data-root data/imported`。原始 prepared manifest 保持不变，其中 Windows 绝对路径仅用于追溯；媒体路径是相对路径。

若服务器也无法访问 judge API，可把生成的完整 run 目录传回本地，通过 `judge --data-root <本地同一数据目录>` 在本地判分。需保持 prepared manifest、图片和运行快照一致；如果在服务器重新执行 prepare 产生新 manifest，应将该 prepared 快照一并传回。

## Git 中转的适用范围

普通 Git 技术上可传图片，但大文件与后续版本会进入仓库历史。GitHub 对普通 Git 的单文件上限为 100 MiB；超过 50 MiB 有警告。压缩包可能超过单文件上限，实际大小以本地下载后统计为准。[GitHub 文件大小说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

Git LFS 使用指针与单独存储，仍需要服务器能访问 LFS 对象服务，并不能自动解决数据源网络限制。[Git LFS 说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

如果只能通过 GitHub 中转，下载后再根据单文件大小选择独立数据快照或分块方案；不要直接强制暂存当前 `data/`，以免缓存、日志和重复归档一并进入代码仓库。
