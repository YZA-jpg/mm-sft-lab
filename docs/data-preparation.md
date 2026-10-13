# SIUO / MSSBench 数据准备

这一批只完成 baseline 的数据准备：下载官方数据、固定版本、检查图片与标注、生成统一输入和独立评分标签。推理 prompt、模型加载和 judge 在数据校验通过后接入。

## 来源与版本

| 内容 | 官方来源 | 下载方式 |
| --- | --- | --- |
| SIUO generation / MCQA 标注 | [sinwang20/SIUO](https://github.com/sinwang20/SIUO) | 固定 Git commit 的两个 JSON 与 README |
| SIUO 图片 | [sinwang/SIUO](https://huggingface.co/datasets/sinwang/SIUO) | 固定 commit 的完整数据快照 |
| MSSBench 标注与图片 | [kzhou35/mssbench](https://huggingface.co/datasets/kzhou35/mssbench)，入口见 [官方仓库](https://github.com/UCSB-AI/MSSBench) | 固定 commit 的完整数据快照 |

首次运行先解析三个来源的完整 commit SHA，将它们写入 `sources.lock.json`，再开始下载文件。重跑默认复用这个锁；需要换版本时使用新的数据目录，避免将两版数据混在一起。

GitHub 标注下载使用固定 SHA 的 raw 链接；GitHub 分支通过 `git ls-remote` 解析。图片下载采用 Hugging Face 官方 [snapshot_download](https://huggingface.co/docs/huggingface_hub/guides/download) 接口及其本地缓存。中断后可在相同目录重跑下载命令。保留快照中的 README、许可证及缓存文件。

## 本地与服务器分工

本地由助手编辑代码、运行离线检查；用户提交并推送。服务器由用户克隆或拉取该提交、安装依赖、下载数据并反馈结果。以下服务器命令在确定服务器仓库目录后逐步执行；上一阶段失败时先保留输出并排查。

数据准备需要 Git、Python 3.10 或更高版本，不需要 GPU。建议使用独立 `.venv-data`，后续模型推理环境单独配置。

服务器 Bash，在仓库根目录创建环境：

```bash
(
  set -e
  python3 --version
  git rev-parse HEAD
  python3 -m venv .venv-data
  .venv-data/bin/python -m pip install -r requirements-data.txt
  .venv-data/bin/python -m unittest discover -s tests -v
)
```

预期：依赖安装完成，离线测试显示 `OK`。此时尚未下载完整 benchmark。

确认环境成功后，服务器 Bash 下载到仓库的 `data/` 目录：

```bash
(
  set -e
  .venv-data/bin/python scripts/download_datasets.py --data-root data
  .venv-data/bin/python -m pip freeze > data/environment-data.txt
  git rev-parse HEAD > data/code-commit.txt
)
```

预期：打印三个 pinned SHA，最终显示 `Download complete`，生成 `sources.lock.json` 和 `download_manifest.json`。真实样本数量由下一步校验统计，不预填固定值。

下载成功后，服务器 Bash 规范化与校验：

```bash
(
  set -e
  .venv-data/bin/python scripts/prepare_datasets.py --data-root data
  git status --short
)
```

预期：打印 `status: validated`、`provenance_status: locked`、场景数、实际推理条数、配对数和唯一图片数。默认 `data/` 和 `.venv-data/` 已被 Git 忽略，下载本身不应产生待提交文件。请反馈这段统计及报错，之后再接模型推理。

若数据应放到其他磁盘，将两条脚本命令的 `--data-root data` 一起改为同一绝对路径；环境和代码版本记录也放到该数据目录。目录须由用户确认有写权限，当前不预设 `/data1` 下的共享目录。

## 文件布局

```text
data/
  sources.lock.json
  download_manifest.json
  environment-data.txt             # 由上面的服务器命令生成
  code-commit.txt                  # 由上面的服务器命令生成
  raw/
    siuo/
      annotations/
        siuo_gen.json
        siuo_mcqa.json
        UPSTREAM_README.md
      images/                     # Hugging Face 原始快照目录结构
    mssbench/
      combined.json               # 同时兼容旧脚本中的 combine.json
      chat/
      embodied/
  prepared/
    inputs.jsonl
    evaluation.jsonl
    manifest.json
```

`media_paths` 相对于 `--data-root`，使用 `/` 分隔，方便在 Windows/Linux 之间迁移。移动数据目录后需要以新目录为基准解析图片；manifest 中的原始绝对路径仅用于追溯。

## 输入和标注

`inputs.jsonl` 每行对应一次待推理请求，保存 `id`、`benchmark`、`task`、来源场景 ID、`query` 和 `media_paths`；SIUO MCQA 还保留原始 `options`。此阶段尚未拼接 chat prompt。

`evaluation.jsonl` 通过同一个 `id` 对齐，保存 category、SIUO 参考答案与风险说明、MCQA 正确选项，或 MSSBench 的 safe/unsafe 条件。官方原始标注完整保留在 `raw/`，后续 judge 可通过来源场景定位其他字段。

推理适配器只把 query、必要的选项及图片内容发给模型。来源路径、样本 ID 和评分标注是程序元数据，不能拼入模型对话。

MSSBench 按官方 [infer_on_data.py](https://github.com/UCSB-AI/MSSBench/blob/main/utils/infer_on_data.py) 的 IF 设置展开：

- chat：每个 `queries` 元素分别配 safe/unsafe 两张图，保留同一 query。
- embodied：按索引配对 `safe_instructions` 与 `unsafe_instructions`，分别保留两侧的原始指令，不假设文本相同。
- `pair_id` 连接两侧，输入 ID 后缀为 `a` / `b`；条件保存在独立评分标注中。
- 场景数与推理条数分别统计，列表展开后的每一条均保留，不因空值或缺图静默跳过。

SIUO generation 与 MCQA 作为独立任务保留全部样本；manifest 报告两者 ID 重合情况与输入差异，不自动删题或改写题目。

## 校验与追溯

规范化脚本检查：

- 标注结构、必需文本、重复 question ID、MCQA 正确选项是否存在。
- 图片路径是否有效、图片能否完整解码；记录大小、尺寸、模式和 SHA-256。
- SIUO 图片若位于快照子目录，通过唯一文件名查找；同名歧义会报错。
- MSSBench 两侧图片必须不同，embodied 两侧指令列表长度必须一致。
- `combined.json` / `combine.json` 同时存在时内容必须一致。
- source lock 与下载记录是否对应，标注 SHA-256 是否被修改。

全部验证完成才替换规范化文件，最后写 manifest。manifest 记录源版本、标注与图片哈希、统计及两个 JSONL 的哈希；后续读取规范化文件应核对其哈希。

若两份版本记录都不存在，允许处理手工准备的数据，但明确标为 `manual_unlocked`。只有一份版本记录、下载未完成或已锁定标注被修改时会失败，不能作为已锁定 baseline 使用。HF 来源 commit 标明下载版本，图片哈希标明本次实际评测资产，二者都应保留。

## 本地验证范围

```powershell
python -m unittest discover -s tests -v
```

离线测试使用临时生成的小图片和合成标注；网络下载通过模拟接口验证版本锁定和调用参数，不下载完整官方数据。它验证脚本行为，不能代替服务器上的真实数据下载与全量校验。

本次不修改 benchmark 内容；未来 SFT 数据配方放入独立目录和配置，避免覆盖 baseline 原始快照。
