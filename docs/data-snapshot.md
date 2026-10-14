# 已校验的 baseline 数据快照

2026-10-14，本地下载与规范化完成，状态 `validated` / `locked`。约 441 MB 的离线包通过 SCP 上传，服务器 SHA-256 校验通过，解压到 `/data1/yaoziao/mm-sft-lab/data/imported`；服务器复核状态与全部统计均与本地一致，复核输出为 `data/imported/server-check`。

| 来源 | 固定 commit |
| --- | --- |
| SIUO 标注 `sinwang20/SIUO` | `18974b65d238ad636d65d238541c7d75279ebb3e` |
| SIUO 图片 `sinwang/SIUO` | `024e80a01795376b9fed12f8073a12f2275f22ee` |
| MSSBench `kzhou35/mssbench` | `12396d24157427b671d97f23a2e370fca98fdf55` |

| 任务 | 原始标注/场景数 | 展开后推理请求数 |
| --- | ---: | ---: |
| SIUO generation | 167 | 167 |
| SIUO MCQA | 167 | 167 |
| MSSBench chat | 300 | 1,200 |
| MSSBench embodied | 76 | 760 |

单回答配置合计 2,294 条请求；MSSBench 合计 980 对；参与评测的图片文件数为 919。SIUO 两个任务共有 167 个问题 ID，输入问题与图片一致，没有仅存在于单个任务的 ID。

以数据目录中的 `sources.lock.json`、`download_manifest.json` 和 `prepared/manifest.json` 为精确追溯记录；本页仅记录已确认版本与统计，不包含数据内容。

传输流程见 [本地准备与离线传输](offline-data-transfer.md)。模型推理与 judge 尚未运行，此统计不代表 baseline 分数。
