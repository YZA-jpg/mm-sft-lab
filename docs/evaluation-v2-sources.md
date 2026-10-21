# Evaluation v2 来源与适配记录

核对日期：2026-10-21。网页核对用于确定入口和原生字段，不等于已在本机下载。实际执行下载时固定当时的 commit/revision，保存 source.lock.json、下载清单 SHA256 和 prepared manifest；不使用未下载却虚构的固定 SHA。

| 来源 | 官方入口 | 本项目处理 |
| --- | --- | --- |
| SIUO / MSSBench / WildJailbreak eval | 沿用 `data/step1/prepared` 锁定快照 | 校验 inputs/evaluation 与图片 SHA；只取 SIUO gen，不把 MCQA 混入 ASR；保留 MSS 配对和两模块；Wild 保留 harmful/benign |
| FigStep | [官方仓库](https://github.com/CryptoAILab/FigStep)、[SafeBench CSV](https://github.com/CryptoAILab/FigStep/blob/main/data/question/safebench.csv) | 下载锁定 commit 的源码 ZIP，仅提取 SafeBench 图片/CSV/许可；固定图片 variant 6；原图字节不改，标准 FigStep 文本 query |
| AdaShield | [system_prompts.py](https://github.com/SaFo-Lab/AdaShield/blob/main/system_prompts.py) | 锁定 commit，AST 提取十类规则字符串；校验 CSV 类别 ID/name；使用用户原文二分类 judge 模板，输出只接受 Result: [[safe]] / [[unsafe]] |
| HoliSafe | [官方评测脚本](https://github.com/youngwanLEE/holisafe/blob/main/eval/eval_by_openai_parallel_v2.py)、[HF benchmark](https://huggingface.co/datasets/etri-vilab/holisafe-bench) | gated 访问由用户申请；只下载 benchmark；原生 JSON+images 或 Parquet；保留 type/category/subcategory/image_safe 等元数据；官方提示 AST 提取 |
| WildGuard | [官方库](https://github.com/allenai/wildguard)、[HF 模型](https://huggingface.co/allenai/wildguard)、[原始模板](https://github.com/allenai/wildguard/blob/main/wildguard/utils.py) | 下载本地 7B checkpoint；锁定模板，用 `/v1/completions` 的原始 prompt，避免重复包 chat template；三项 yes/no 严格解析 |
| MMStar | [官方仓库](https://github.com/MMStar-Benchmark/MMStar)、[HF](https://huggingface.co/datasets/Lin-Chen/MMStar) | 原生 Parquet，保留 question/options、answer、category 和 image bytes；本地选项匹配，不自动猜选项、不调用额外 LLM |
| MathVista | [官方库](https://github.com/lupantech/MathVista)、[HF](https://huggingface.co/datasets/AI4Math/MathVista)、[抽取提示](https://github.com/lupantech/MathVista/blob/main/evaluation/prompts/ext_ans.py)、[归一化](https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py) | 只下载 testmini Parquet，answer_type/precision/choices 保留；文本 judge 抽取，不传测试答案；类型/精度本地比较 |
| MMVet | [原始 v1 库](https://github.com/yuweihao/MM-Vet)、[v1 发布包](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip)、[官方 evaluator notebook](https://github.com/yuweihao/MM-Vet/blob/main/mm-vet_evaluator.ipynb) | 原始 v1 的 218 问，不偷偷替换 v2；AST 提取 notebook 中评分提示；score 严格 0.0–1.0 十分位 |
| vLLM | [Qwen3-VL recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html) | 使用 OpenAI 兼容服务；独立 Linux 环境，>=0.11.0 是最低版本，不是服务器已经验收的锁版本 |

## 与官方结果的差异，汇报时必须说明

1. 云 judge 统一使用用户账户已验证可传图的 DashScope DeepSeek；这不同于相关论文所用 GPT/Claude 等评审模型。接口连通性不等于 judge 准确率验证。
2. SIUO 暂定 ASR@3 any-success，不沿用旧多次多数票；FigStep ASR@5 同理。这是用户新主表的暂定解释，正式冻结前需要确认。
3. FigStep judge 的 scenario 按类别、scenario_rule 按 AdaShield 对应规则填入，单独 user message 传被测回答。原始具体问题留在 reference，不作为额外 judge 字段；是否期望 scenario 填完整具体问题也列为待确认。
4. 不沿用 AdaShield 解析失败时默认 safe 的兜底；格式错误不生成安全标签，不进入有效 y。
5. HoliSafe 按官方当前实现使用 SSS RR、其他类型 ASR，False / 有效标签数；同时保留 adherence。SSS 失败既可包含拒答，也可包含严重不相关等行为，不将它过度解释成精确拒答分类器。
6. WildJailbreak 使用 WildGuard，不再使用旧 DeepSeek 的二项 JSON；模板和权重身份都记录，解析失败独立统计。
7. MathVista 使用官方 few-shot 文本，但适配严格 JSON `{"answer": "..."}` 或 null；不是逐字复现官方自由文本抽取器。choice 编辑距离、integer/float 和 precision 归一化基于官方方法。MMStar 为本地严格抽取，可能比论文的宽松抽取少接受一些回答形式。
8. MMVet 默认 judge 一次，参数支持多次；论文多次评分设置需要另行确认。目前报告每题均分和总分，不声称已复现论文的多次标准差。
9. 所有角色都省略生成参数。服务默认输出上限、采样、thinking 等仍存在；和论文固定参数不同，必须记录 vLLM/模型默认配置及终止原因。ASR@k 的重复可能相同。
10. 用户于 2026-10-21 明确本轮不做训练数据重叠检查，直接推进 baseline；该检查不作为启动前置步骤。既有训练数据和历史过滤结果保留，不新增去重或排除规则，也不据此声称新增全套已经完成去污染。

## 数据不变性与许可记录

图片只做完整性校验和按内容去重，保留原 bytes，不缩放、不重编码、不针对审核失败裁图。prepared 的原图 hash 与请求内图像 bytes 对应。第三方源码只做 AST literal 提取，不 import/exec 其脚本；ZIP 拒绝路径穿越、符号链接和重复成员。

上游许可随来源一起下载/保留，代码与图片/数据许可可能不同，以各来源当时的许可和访问要求为准。下载资料留在被 Git 忽略的 data 目录，服务器运行只需 prepared、WildGuard 权重和代码。发布实验材料时应保留来源与许可，不能把本项目许可当成第三方数据统一许可。
