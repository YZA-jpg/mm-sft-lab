# 全量生成中断：WebP MIME 兼容与恢复说明

> 文档状态（2026-10-23）：历史异常及原代码兼容恢复方案。当前已修改核心 MIME 和调度，旧运行不能用新版代码直接续跑；按 [当前手册](baseline-runbook.md) 新建 smoke / 全量。下方恢复命令仅用于原版本追溯。

记录日期：2026-10-23。用户执行服务器命令；本次助手只完成本地排查、代码、离线测试和指令准备。

## 1. 用户回传的事实

原目录 runs/B0-v2-64k-full-20261022T092751Z，指纹 c9cc68b995ee533ef30072ba368928d0f52e659f4a51e46966a3cffa92146d18。成功 500/2461，另有 mssbench/chat/0031/0/b#g0 的 local_preparation_error；1960 个 job 尚未尝试。主进程已退出，generation 无锁且无在途流。此时是异常停止，不是仍在后台慢慢生成。

最后完成记录 epoch=1792688839.559722，即北京时间 2026-10-23 01:07:19；用户状态快照 UTC 04:07:42，即北京时间 12:07:42。两个时刻相差约 11 小时；不能将这段停机空档计作有效推理时间。开始目录时间到最后完成约 7.66 小时，是一段粗略观测时长，不代替阶段计时。

500 份有效回答中 stop 454、length 46，截断率 46/500=9.2%。若按每份 65536-token 额度计算，仅 46 份长回答就约 3,014,656 个最终输出 token，另外还存在正常回答和可能的失败尝试成本。恢复预检会读取真实 usage，分别打印最终有效回答的已知 token 总量和 length 部分，不把局部估算冒充完整成本。

## 2. 已在本地复现的错误，服务器仍由预检核对

同源锁定图片 assets/50ca8fb72738e6d40a8e5f7f180aedd7b19387566fcc6ed945f342e4e1a97314.webp，53584 bytes、WEBP、1024×683。原 evaluation_v2.prompts._image_url 依赖系统 mimetypes.guess_type；本地返回 None，随后构造 data URI 时触发 TypeError: can only concatenate str (not "NoneType") to str。异常发生在 HTTP 请求之前。此处是此前完整消息构造预检未覆盖到的兼容缺失，不是图片缺失、模型拒答或 DeepSeek 问题。

真实本地快照共有 PNG 281、JPEG 602、WebP 5 张；当前系统 MIME 表缺失的仅为这 5 张 WebP。显式注册 image/webp 后，全部 2127 个原始请求（对应 2461 个逻辑生成 job）的 JSON/UTF-8 请求构造离线通过。未缩放、转码、修改图像或问题。真实检查产物留在 runs/local-webp-preflight-20261023.json，不上传数据到 Git。

用户尚未提供服务器失败行的 error 字段，不能直接把本地复现等同于服务器确诊。恢复入口首先读取服务器原始错误：只有 WebP 输入上的同一 NoneType/str 拼接错误且该失败 job 没有 HTTP 尝试，才允许修复；磁盘、网络、其他异常会显示原错误并以非零码退出，不启动模型。

## 3. 恢复方式和身份保存

新增 scripts/resume_eval_v2_webp.py，原 evaluation_v2/*.py 保持字节不变，原指纹检查保持启用。恢复进程注册 WebP MIME 元数据后，逐条重建已有成功回答的完整 HTTP 请求体并核对 request_sha256，同时核对 response_sha256；任一不符则停止。它不是新评分协议，不改变输出预算、采样、模型、数据、并发、GPU、重试次数或 trial。

preflight 通过后，recovery/webp-<UTC时间>/receipt.json 保存修复脚本 SHA、原指纹、MIME 前后值、成功请求核对数、结果日志原始 SHA、成功响应摘要、待处理数和 PID 记录；resolved-config.json 是原 manifest 解析配置的副本。随后调用原 CLI 的 --resume --generate-only，模型与运行参数均从原 manifest 恢复，保留 venv Python 的原始路径，不能将其 resolve 成另一个解释器。

在当前记录下，预期跳过 500 份成功回答，修复 1 条准备失败，继续 1960 条未尝试，总待处理 1961。失败历史不删除，原始回答不重采样。该 MIME 准备失败未消耗 HTTP 次数，因此修复后第一份 HTTP 尝试仍是 number=1；后续网络重试仍遵守每 job 初次加五次的全局预算。新日志追加到 job.log，job.pid 更新为续跑主 PID，原 PID 记录保存在预检 receipt；旧监控命令继续可用。

生成阶段原有已完成 sessions 和 HTTP/流式尝试计时仍保留，恢复后报表从全部历史汇总。原 pipeline 因错误缺少 completed 事件，不能仅把 pipeline_wall_seconds_completed_sessions 当作含此次失败的总耗时；旧会话会列入 interrupted_pipeline_sessions。生成阶段 wall_seconds_completed_sessions 包含本次失败前已经写出的阶段会话。模型加载、设备采样和重试成本也须按原记录解释。

## 4. 本地验证

共 69 项 v2 离线测试通过，其中 9 项新增恢复回归验证：复现缺失 MIME、预检无模型调用、原图字节不变、已有成功请求及原指纹不变、历史保留且成功项不重跑、拒绝无关错误、拒绝活动锁、从原 manifest 恢复参数。另有全部真实 2127 条请求构造通过。未执行 SSH、模型/GPU/API 调用或 Git 提交；服务器根因和恢复结果以用户回传为准。

## 5. 本地 Windows PowerShell：仅提交相关文件

```powershell
Set-Location -LiteralPath "D:\Desktop\AI Safety\mm-sft-lab"

$files = @(
    "scripts/resume_eval_v2_webp.py",
    "tests/test_eval_v2_webp_recovery.py",
    "docs/plans/2026-10-23-webp-mime-recovery.md",
    "docs/webp-preparation-incident.md",
    "docs/data-and-experiment-notes.md",
    "docs/siuo-mss-64k-full-run.md",
    "docs/plans/2026-10-22-target-output-64k.md"
)
git add -- $files
if ($LASTEXITCODE -ne 0) { throw "暂存失败" }
git commit -m "Add audited WebP recovery and baseline run notes"
if ($LASTEXITCODE -ne 0) { throw "提交失败" }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "推送失败" }
```

## 6. 服务器 Bash：核对后在原目录后台恢复

```bash
(
  set -euo pipefail
  cd /data1/yaoziao/mm-sft-lab
  git pull --ff-only

  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1
  task_run="$(cat runs/latest-B0-v2-64k-full.txt)"

  .venv-data/bin/python scripts/resume_eval_v2_webp.py \
    --run-dir "$task_run" --check-only

  free_mib="$(nvidia-smi -i 1 \
    --query-gpu=memory.free --format=csv,noheader,nounits \
    | tr -d '[:space:]')"
  if [ "$free_mib" -lt 55000 ]; then
    echo "GPU 1 当前空闲 $free_mib MiB，暂不启动。"
    exit 1
  fi

  nohup .venv-data/bin/python -u scripts/resume_eval_v2_webp.py \
    --run-dir "$task_run" \
    >> "$task_run/job.log" 2>&1 < /dev/null &

  task_pid=$!
  printf '%s\n' "$task_pid" > "$task_run/job.pid"
  sleep 1
  if ! kill -0 "$task_pid" 2>/dev/null; then
    tail -n 40 "$task_run/job.log"
    exit 1
  fi
  printf '续跑主进程存活，PID=%s\n原任务目录：%s\n' "$task_pid" "$task_run"
)
```

set -e 保证 check-only 未通过时后面的 GPU 启动不会执行。如果输出 status=blocked，请保留并回传 error，不直接跳过预检。GPU 1 显存不足时不改原运行的 GPU 身份、不停止其他用户进程；等待资源或另行制定迁移方案。

恢复后仍使用已有 latest-B0-v2-64k-full.txt、job.pid、job.log 和 evaluate_v2.py status 查看。应先看到兼容预检汇总、vLLM target ready，然后出现新的 generation 记录；不会重跑 500 份成功回答。length 计入有效生成及截断率，后续 judge 继续按原来的 x/y、覆盖率、拒答/格式错误排除和各 benchmark 计时规则执行。

