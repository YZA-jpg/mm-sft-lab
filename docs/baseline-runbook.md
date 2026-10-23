# 当前 baseline 执行手册

更新：2026-10-23。先验证修复，再单独启动全量；暂缓训练和学长确认。本手册替代旧混合调度全量指令，旧运行文件保留。

## 本轮固定设置

| 范围 | SIUO gen | MSSBench chat | MSSBench embodied | 总生成次数 |
| --- | ---: | ---: | ---: | ---: |
| 新 smoke | 1 题 × 3 | 5 配对 × 2 条 × 1 | 1 配对 × 2 条 × 1 | 15 |
| 新全量 | 167 题 × 3 | 1200 条 × 1 | 760 条 × 1 | 2461 |

smoke 的五个 chat 配对来自场景 0031、0045、0067、0072、0297，覆盖全部 5 张原始 WebP，包含此前失败的 `mssbench/chat/0031/0/b`。原问题和图片 bytes 不变，配对不拆分。MSSBench 全量 980 配对、safe/unsafe 各 980 条。

数据目录沿用 `data/eval-v2/siuo-mss-stream-v1`，不重新准备或覆盖。其他六项 benchmark 的数据/服务器验收后续另行进行；本次全量指上述两个 benchmark。

单卡 vLLM：**SIUO 完成后开始 MSSBench**，单数据集内最多 4 并发。MSSBench chat/embodied 内部仍交错，分别保留计数和请求成本，但只有整个 MSSBench 的时间是独立数据集墙钟。模型只加载一次，共享启动/关闭开销另记。

目标仅发送 `max_tokens=65536`，不覆盖温度、top_p、top_k、seed、thinking 等默认值。DeepSeek/WildGuard 仍不发送解码参数，本轮启动不调用它们。达到上限属于有效但截断的回答，完整保存并计数，不自动重试或追加 trial。MIME 修复解决请求准备异常，不保证模型停止重复；单条仍可能花约 20 分钟或更久。

## 1. 服务器 Bash：拉取、离线检查、启动新 smoke

先由用户在本地提交、推送本轮文件，再执行：

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  git pull --ff-only
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1

  .venv-data/bin/python -m unittest discover -s tests -p 'test_eval_v2_*.py'
  .venv-data/bin/python scripts/start_fixed_baseline.py smoke --check-only
  .venv-data/bin/python scripts/start_fixed_baseline.py smoke --gpu 1
)
```

预期离线计划：15 次、WebP 5 张、SIUO 3 / MSSBench 12。启动器会再次完整检查请求，然后检测 GPU 1 至少空闲 55000 MiB、端口 18014 未占用，再创建后台独立会话。显存不足时停止；可将 `--gpu 1` 换成经检测有空间的卡。不会停止其他进程。模型服务使用 memory=0.65、KV cache=40 GiB；当次空闲检测不是显存预留。

后台任务保存 PID、日志、启动计划、Git 提交号和目录指针，关闭终端后继续。`detached_generation_started` 只表示后台进程已创建，需继续确认模型就绪和结果。若预检失败或模型启动失败，保留错误并反馈，不跳过。

未来更换模型可加 `--model-path /path/to/checkpoint --model-label T1`；LoRA 使用完整 base 路径并加 `--adapter-path /path/to/adapter`。每个模型/adapter 均需自己的 smoke。默认 B0 模型为 `/data1/sharehub/models/Qwen3-VL-4B-Instruct`。

## 2. 服务器 Bash：检查 smoke

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  test -s runs/latest-B0-v2-fixed-smoke.txt
  task_run="$(cat runs/latest-B0-v2-fixed-smoke.txt)"
  printf '任务目录：%s\n' "$task_run"
  ps -p "$(cat "$task_run/job.pid")" -o pid,etime,stat,args || true
  tail -n 25 "$task_run/job.log"
  if [ -f "$task_run/manifest.json" ]; then
    .venv-data/bin/python scripts/evaluate_v2.py status --run-dir "$task_run"
  fi
)
```

启动日志出现 `vLLM target ready` 后才进入生成。预期结束时 generation valid=15、remaining=0、无准备/网络等终态错误，5 张 WebP 均有结果。`length` 单列，不视作请求失败。smoke 结束后先回传日志摘要，检查完整回答与计时，再手动决定启动全量。

如果 manifest 暂未出现，先看 `job.log`；启动器完整预检和环境读取尚可能进行中。模型启动详细日志在任务目录 `services/target-*/server.log`。不能只凭无进程就判断成功，需要看生成数量和错误。

## 3. smoke 验收后：单独启动新全量

**不与上面的 smoke 命令一起执行。**

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  test -s runs/latest-B0-v2-fixed-smoke.txt
  smoke_run="$(cat runs/latest-B0-v2-fixed-smoke.txt)"
  .venv-data/bin/python scripts/start_fixed_baseline.py full \
    --gpu 1 --smoke-run "$smoke_run"
)
```

启动器核对 smoke 的代码/数据/模型/配置、15 个有效结果和非重叠数据集计时；不接受 mock、错误、旧混合 smoke 或其他模型的 smoke。全量必须是新目录，不将旧 500 条结果混入。指针为 `runs/latest-B0-v2-fixed-full.txt`，监控命令将上一节 smoke 指针替换为此 full 指针即可。

本阶段仍是生成-only。生成结束后单独做 judge：网络类首次加 5 次重试；护栏不重试；格式、鉴权、输入限制、judge 拒答分开记录。五次网络重试不意味着所有非护栏缺失都会自动补齐。判分补偿绑定同一份原回答，不能重采样目标模型来补标签。

## 4. 逐数据集时间和产物

```bash
cd /data1/yaoziao/mm-sft-lab
task_run="$(cat runs/latest-B0-v2-fixed-smoke.txt)"
.venv-data/bin/python scripts/export_eval_v2_timing.py --run-dir "$task_run"
```

full 则替换指针。运行中也可只读导出，但不会统计尚未结束的请求。每次写新 `timing-snapshot-*` 目录，不覆盖原始文件。

| 路径（相对任务目录） | 内容 |
| --- | --- |
| `manifest.json`、`selection.jsonl`、`dataset-manifest.json` | 冻结模型/数据/代码/配置指纹、样本与参考快照 |
| `request-preflight.json` | 全选中请求离线构造、MIME 数量和 WebP 样本清单 |
| `generation/results.jsonl` | 完整回答、finish reason、usage、尝试、请求/响应 hash、逐条耗时 |
| `generation/streams/*/attempt-*` | 流事件、进度、原始 / 部分回答、成功和失败尝试文件 |
| `generation/transport-attempts.jsonl` | HTTP 尝试预算及时间；首次加最多五次网络重试 |
| `generation/benchmark-sessions.jsonl` | SIUO/MSSBench 起止边界、并发、独立运行秒数及失败标记 |
| `generation/sessions.jsonl`、`pipeline-sessions.jsonl` | 阶段与管线会话，失败/中断空档另计 |
| `services/*/launch.json`、`server.log`、`gpu-samples.jsonl` | 默认生成配置文件 hash、服务启动开销、vLLM 日志、GPU 采样 |
| `timing-snapshot-*/dataset-timing.csv`、`.json` | 数据集 / 子模块、生成 / judge 分阶段数量、时间、重试和截断 |
| `report/` | 当前生成质量及 x/y/coverage 报表；生成-only 不代表已有安全分数 |

正式比较使用 CSV 中 `benchmark=siuo/mssbench`、`split=ALL` 的 `isolated_benchmark_wall_seconds`，包含准备、重试、退避和最终慢请求等待，排除模型共享加载和停机空档；失败/未完成会话必须连同标记汇报。子模块行该字段为空，不误称独立运行耗时。`sum_job_wall_seconds` 是并发请求延迟之和，不当作墙钟；GPU 采样也是共享卡状态，不能当作本任务独占 GPU 时间。

## 历史与当前修复边界

旧运行 `B0-v2-64k-full-20261022T092751Z` 的 500 成功、1 准备错误和 46 条截断保留为诊断。当前核心使用显式 PNG/JPEG/WebP/GIF MIME 映射，启动前检查每个选中消息；旧版恢复脚本仅保留历史用途，当前核心变更后不能对旧指纹直接续跑，不能禁用指纹检查。

当前服务器根因仍以原失败行的 error 为准，本地已明确复现并修复 WebP MIME=None 的拼接异常。新 smoke 用实际服务器端请求验证此修复，助手不把本地通过写作服务器已通过。
