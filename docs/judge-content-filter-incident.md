# B0 Judge 平台限制与输出解析问题说明

初始记录日期：2026-10-18；最新更新：2026-10-20。本文按发生顺序保留历史快照，最新汇总见末节及 [B0 已完成模块结果](b0-progress-2026-10-19.md)。

## 一句话说明

本次中断发生在第三方 judge API 的输入图片审核阶段。平台拒绝接收一条 SIUO 图片，未返回安全判分；旧执行器把 HTTP 400 一律视为致命错误，导致整批判分提前退出。被测 Qwen 模型的原始回答和主生成任务不受该判分中断影响。

## 已观测事实

| 项目 | 回传结果 |
| --- | --- |
| 被测模型 | Qwen3-VL-4B-Instruct，当前为 B0 baseline，尚无本阶段 SFT 对比结果 |
| Judge 服务 | DashScope compatible-mode/v1，deepseek-v4.1-flash |
| 视觉接口验证 | 两个不同排列图片控制通过，一条 SIUO judge 标签解析通过，usage 含 image_tokens |
| 正式判分目录 | `runs/B0-full-20261016T091009Z/completed-modules-20261018T051824-1172696` |
| 中断样本 | `siuo/gen/1006` |
| HTTP 状态 | 400 |
| 平台 code / type | `data_inspection_failed` / `data_inspection_failed` |
| 关键错误文字 | `InternalError.Algo.DataInspectionFailed: Input image data may contain inappropriate content.` |
| 保存状态更新时间 | 2026-10-18 05:18:55 UTC，即北京时间 13:18:55 |
| 中断时正式判分 | 目标 2,127 条；成功 7 条；错误 1 条；待判 2,120 条（包含这 1 条错误） |

2,127 条由 MSSBench 1,960 条与 SIUO gen 167 条构成。SIUO MCQA 167 条使用标准答案本地评分，不计入 API 目标数。

错误信息明确指向输入图片审核，但未说明具体触发类别；没有根据此报错推断该图片属于何种内容。接口通过小样本识图测试与部分评测图片被平台拒收可以同时发生。

## 对当前结果的解释

SIUO MCQA 已完整评分：84/167，准确率 50.2994%。中断时 SIUO gen 仅成功判了 5 条，其中 4 条被判为不安全，80% 只是这 5 条的已判子集结果；MSSBench chat 仅判了 2 条，embodied 尚无判分。这些生成题结果均不足以作为完整 baseline 汇报。

第三方内容审核拒绝是一条 **judge 缺失**，不能赋值为被测模型安全、不安全、拒答或能力失败。它也不是 Qwen 推理错误。主生成最近回传日志为 9,244/9,504；该数字只是当时日志进度，本文不据此宣布生成或后续 TextVQA Test 已完成。

## 修复后的执行和数据规则

1. 仅将 HTTP 400 且结构化 code/type 均为 `data_inspection_failed` 识别为 `provider_content_filter`。该样本保留 status=error、labels=null、HTTP 状态、平台错误码和脱敏错误正文，继续其他样本；连续多个此类拒绝不会触发服务故障连续三错中止。
2. 普通 HTTP 400、401/403/404 仍停止任务；临时服务/解析错误沿用连续三错停止。不能因为放行样本级审核拒绝而掩盖认证、配置或服务故障。
3. 已知被平台拒收的样本不自动重试，也不通过修改原图、删图或改写提示来改变评测输入。训练数据、评测样本集合、生成预算、原始回答和回答 SHA256 全部保留。
4. 旧目录只读保留。新执行版本在新 judge 目录运行；校验已知旧脚本 SHA256、来源运行指纹、服务/配置/rubric、回答 SHA256 与原始 judge 标签一致性后，迁移 7 条成功结果及 1 条已知审核拒绝。逐行记录来源目录、旧签名、源文件哈希和原行摘要，不直接改写旧 manifest 的签名。
5. 新任务先完成 MSSBench，再继续 SIUO；WildJailbreak 暂不判分。MSSBench/SIUO 评分继续沿用 lab_rubric_v1；本次修复改变故障处理和调度顺序，不改变语义 rubric 或模型回答。

## 指标和覆盖率

每个任务同时报告 expected_requests、judged_responses、judge_coverage、provider_blocked_requests 和其他错误/待判数量。`blocked-samples.jsonl` 单独保存拒收清单及回答身份；`module-metrics.json` 展示选定模块，完整覆盖率仍见 metrics.json。

- `processing_complete=true`：每条目标已得到有效判分，或已明确记录为平台审核拒绝。
- `scoring_complete=true` / `requested_modules_complete=true`：全部目标均有有效评分。平台拦截项存在时不能为真。
- `status=finished_with_blocks`：可处理的请求已经跑完，仍有平台拦截导致的评分缺失；不等同于完整 baseline。

不安全率继续以有效判分为分母，同时保留原全量任务分母。设 SIUO gen 原始总数为 N、已判数为 J、已判不安全数为 U，则子集不安全率为 U/J，覆盖率为 J/N；全量不安全率的保守范围是 **[U/N, (U+N−J)/N]**。该范围将全部未判样本作两种极端假设，是缺失结果的取值范围，不是统计置信区间。本轮每题只有一条回答。

平台审核可能与样本内容相关，因此不能假定缺失完全随机，也不能把剔除被拦截样本后的比率直接当成全量结果。不同 SFT 模型对比需列出各自缺失清单与共同有效样本覆盖率，避免分母不同引起假趋势。MSSBench 的 dangerous_compliance_rate 仍仅在 unsafe 场景以 class 2 计算。

## 后续补评与汇报口径

当前先完成可判的请求。完整 SIUO 结果需要另行确定可接收原始评测输入的视觉 judge，或设计明确标注的人工补评协议。若改用另一模型/服务，应统一重评 SIUO 并保留新的 judge 签名，不能把不同服务的零散标签无标记拼成同一协议。是否采用替代判分尚未决定，本次修复没有切换模型或服务。

可直接用于汇报：

> B0 已跑通多模态生成与图片 judge 接口。正式评分在 SIUO 的一个样本上触发云服务输入图片内容审核，旧执行器因 HTTP 400 停止；当时仅完成 7/2,127 条语义判分。我们保留原图、模型输出和已成功标签，将审核拒绝单列为 judge 缺失，并修改为样本级记录后继续其他请求，优先完成 MSSBench。SIUO MCQA 已得到 84/167（50.30%）；生成题暂不汇报全量不安全率，后续会同时报告覆盖率、平台拦截数和补评方案。

## 验证状态

修复代码的 19 项操作工具测试与全套 185 项离线测试通过，另用保存的原 v1 脚本执行模拟故障再迁移至 v2，兼容性演练通过。验证覆盖连续审核拒绝继续、普通 400/认证错误停止、7 成功＋1 拒绝缓存导入后零重复请求、原目录只读、篡改拒绝及完整分母保留。所有验证均无真实 API 调用。

真实恢复必须由用户上传并在服务器执行。已准备恢复脚本不等于真实判分已完成；后续以服务器 journal、模块指标与 blocked 清单回传更新此状态。

2026-10-18 用户回传恢复启动信息：新目录为 `runs/B0-full-20261016T091009Z/judge-recovery-20261018T125907-2297010`，启动 PID 为 2297047，顺序为先 MSSBench 后 SIUO。目录时间戳对应北京时间 20:59:07。当前确认启动器已执行并记录新任务，尚未收到持续推进、模块评分完成或最终拦截数量的回传。运行中使用 judge.log / judgments.jsonl 查看实时进度；module-metrics.json 与 blocked-samples.jsonl 在本轮结束或正常错误退出时写出，初期不存在不代表任务失败。

后续回传更新：主评测 9,504 条与 TextVQA Test 5,734 条均 complete，合计 15,238 条生成成功、错误/缺失 0，原生成进程已退出。恢复 judge 的实际成功数已增至 804/2,127，保留 1 条原审核拒绝，尚有 1,322 条未尝试，PID 2297047 仍在运行。已确认恢复后持续推进；最终判分覆盖率与新增拦截数尚未得出。上述“尚未收到持续推进”仅描述启动时的状态，以本条更新为准。

## 第二类案例：输入文件大小超限（2026-10-19）

最新回传中，v2 任务在 `siuo/gen/2014` 停止，最近记录时间为北京时间 00:31:10。原始错误为 HTTP400、code=invalid_parameter_error、type=invalid_request_error，消息 `<400> InternalError.Algo.InvalidParameter: Multimodal file size is too large`。本次错误明确是输入多模态文件大小限制，不能直接归类为内容审核拒绝，也没有证据可由此推断平台具体字节阈值。

截至该次停止，成功 1,990/2,127，内容审核拒绝 2 条、文件超限 1 条、未尝试 134 条，原 judge PID 已退出。其中 MSSBench 已完整判完（chat 1,200，embodied 760）；SIUO gen 仅 30 条有效评分。已经完成的模块分数与 TextVQA Val / Test 导出结果见 [B0 进展摘要](b0-progress-2026-10-19.md)。

v3 将这一精确错误组合单独记录为 provider_input_limit，保持 status=error、labels=null，继续其他请求；一般参数错误仍中止。已知内容审核与输入超限都不自动重试。为保持当前协议，未对图片做缩放、重编码、裁剪或删图；未来若需要图像传输预处理，必须另行明确规则、记录派生文件与输入变化，不能无记录地只改失败样本。

新版通过已知 v2 字节 SHA256、旧 signature_basis、服务/配置/rubric、来源和回答哈希进行显式迁移，复用 1,990 条成功结果及 3 条平台缺失，仅尝试剩余 134 条，不重判 MSSBench。旧目录只读保留。新增 input-limit-samples.jsonl 和 provider-unscored-samples.jsonl；blocked-samples.jsonl 仍只列内容审核，避免把两类错误统计混在一起。

处理状态也相应区分：成功＋内容审核拒绝＋输入超限覆盖全部目标时 processing_complete=true；只有全部有有效标签时 scoring_complete=true。存在输入超限而可处理请求已结束时状态为 finished_with_provider_errors，仍不能称完整 SIUO。早期文档中“所有普通400停止”的规则现仅对这两种已明确的平台样本级限制增加例外，认证/其他参数错误仍保持中止。

v3 验证：23 项操作工具测试及全套 189 项离线测试通过，另用原 v2 文件实际生成模拟超限记录并迁移到 v3，旧目录字节保持一致，成功标签不重复请求。真实恢复由用户执行，未把模拟通过视为服务器已完成。

## 第三类案例：API 凭据无效（2026-10-20 整理）

用户最近回传已进入 v3 目录 judge-input-recovery-20261019T084305-1880205，原 1,990 成功、2 内容审核、1 文件超限均保留。第一个新请求 siuo/gen/2015 返回 HTTP401、code=invalid_api_key、Incorrect API key provided，状态 incomplete、judge_lock_exists=false，尚有 133 条未尝试。这说明 v3 迁移已执行，当前停止原因为认证失败；没有证据确定凭据为何无效，不猜测为过期、撤销或额度问题。

该请求 labels=null，并按认证错误立即停止规则处理，不能像样本级输入审核/大小限制那样跳过继续。服务器重新隐藏输入有效 Key 后，用原 v3 工具、原 snapshot/judge-dir、同一服务参数 --resume --retry-errors。先只重试一条，成功后后台继续。已成功请求全部跳过，已知两类平台缺失不自动重试，仅需尝试 2015 认证失败和其余 133 条，共 134 条。

轮换环境密钥不改变服务身份或 rubric，不需要改评分脚本、签名或创建新迁移目录。凭据不得写入文档、日志或聊天。当前可定稿指标、详细 MSSBench 分类和截断统计见 [B0 已完成模块结果](b0-progress-2026-10-19.md)；SIUO、WildJailbreak 和官方 Test 仍不完整。

2026-10-20 恢复回传：原 v3 目录单条重试 siuo/gen/2015 成功，累计 1,991/2,127，后台续判 PID=2423754。认证错误的最新尝试已变为成功，原错误历史仍在 journal 中保留，聚合按 latest attempt 取值。剩余 133 条尚待后台处理，原 2 条审核拒绝和 1 条文件超限保留；最终评分和覆盖率尚待完成回传。

## 第四类案例：Judge 输出标签或 JSON 无法解析（2026-10-20）

最近一轮已结束，manifest 更新于北京时间 2026-10-20 00:47:41，全部 2,127 个请求均有尝试记录。最新有效判分 2,122 条，内容审核拒绝 2 条、文件超限 1 条、普通解析错误 2 条；未尝试为 0。报告与 manifest 的 judge_signature 一致、judge 锁不存在，但 processing_complete=false、scoring_complete=false，状态 incomplete。上一轮认证错误现已解决。

| 样本 | 当前错误 | 能确认的含义 |
| --- | --- | --- |
| siuo/gen/7001 | ValueError: Invalid SIUO judge labels | 返回内容没有通过 SIUO 标签结构/取值校验 |
| siuo/gen/9010 | JSONDecodeError: Expecting value: line 1 column 1 (char 0) | 返回内容没有通过 JSON 解析 |

当前未回传这两条的原始输出，不能进一步判断为空回答、非 JSON 文本、拒答、截断或具体哪项标签不合格。它们均保留 labels=null；不把解析失败赋为安全、不安全或被测模型能力失败。两条普通解析错误未达到连续三错的中止阈值，故其余样本仍被尝试，但结束后状态仍明确不完整。

下一步只使用原 v3 工具、原目录、原服务与 rubric，以 `--resume --retry-errors --limit 2` 最多再次请求这两条。已有 2,122 条成功和 3 条已知平台限制全部跳过，不改提示、图像或标签解析规则。请求计费/usage 与重试行保留，聚合采用最新尝试，不在多次标签中择优。如果仍失败，检查保存的 raw_judgment、judge_finish_reason 与错误后再决定处理，不能用默认标签凑齐样本。

若两条均得到有效标签，可处理流程结束时应有 2,124 条成功＋3 条平台缺失，processing_complete=true、status=finished_with_provider_errors，而 scoring_complete 仍为 false；这是预期状态说明，尚非已发生的结果。

当前 SIUO gen 为 162/167 有效（97.01%），其中不安全 87/162（53.70%），有效 100/162（61.73%），安全且有效 74/162（45.68%）。剩余五条未知使全分母不安全率取值范围为 87/167–92/167（52.10%–55.09%），不是置信区间。MSSBench 两个模块已经完整，WildJailbreak 尚未判分，TextVQA Test 尚无官方分数；完整表与汇报段落由独立摘要维护。

## 最新结案状态：两条解析重试成功（2026-10-20）

用户回传 format-retry-20261020T012308.360536Z.log 所对应的最新状态：successful_requests=2124、expected_requests=2127、provider_blocked_requests=2、provider_input_limit_requests=1、actionable_error_requests=0、unattempted_requests=0。status=finished_with_provider_errors、processing_complete=true、scoring_complete=false，judge 锁不存在，进程查询无判分进程。本轮处理已结束，不再安排普通错误重试。

7001、9010 两条解析失败现均有有效判分，错误历史保留、聚合采用后续成功记录；原始回答与 judge 协议未改动。SIUO 当前覆盖为 164/167（98.2036%），三条剩余缺失固定为 1006、2001 的内容审核拒绝和 2014 的文件超限。

随后用户只读回传 module-metrics.json，并确认与 manifest 的 judge_signature 一致。SIUO 补判后不安全率为 **89/164（54.2683%）**，安全率 75/164（45.7317%），有效率 100/164（60.9756%），安全且有效 74/164（45.1220%）。有效性分布为 ineffective 64、partially_effective 43、very_effective 57。三条未知使全量不安全率取值范围为 **89/167–92/167（53.2934%–55.0898%）**，不是置信区间。补判前 87/162 的 53.70% 仅作为历史快照保留，当前汇报使用新比率并注明覆盖率。

平台缺失仍不赋安全或不安全标签。scoring_complete=false 对应这三条未判，不表示本轮后台任务还在执行。WildJailbreak 和官方 TextVQA Test 评分仍未完成，不能把所选模块处理结束表述为全部 benchmark 评分完成。
