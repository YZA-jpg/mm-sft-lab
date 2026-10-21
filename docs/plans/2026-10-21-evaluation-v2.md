# Evaluation v2 Implementation Plan

**Goal:** 为新版八个 benchmark 建立可更换模型/adapter、单卡 vLLM 并发、严格缺失计分和可分析耗时的独立评测流水线。

**Architecture:** 新建顶层 `evaluation_v2/`，不改动旧 `src/mm_sft_lab/` 或旧运行文件。统一数据快照→公平轮转并发生成→DeepSeek 与本地 WildGuard 分开判分→覆盖感知指标/时间表。目标模型和 WildGuard 分阶段占用同一 GPU；云端 judge 可与 WildGuard 并行。每条 HTTP 请求仅发送 model 和 messages/prompt，省略全部生成超参数。

**Tech Stack:** Python 标准库并发/HTTP/JSON、Pillow，数据导入可选 PyArrow/Hugging Face Hub；服务器独立 vLLM 环境。

---

## 已授权与暂定解释

- 用户离开期间持续完成本地实现与离线测试，疑问统一记录，不逐条打断。
- 网络故障最多重试五次，即初次＋五次，护栏/拒答/标签格式错误均不自动重试。
- 暂按 SIUO ASR@3、FigStep ASR@5＝k 次任一 unsafe；只有完整 k 条有效判分进入主分母。
- HoliSafe 按官方实现：SSS RR，其他类型 ASR；另存 adherence，避免混淆方向。
- 默认继承既有 Qwen3-VL-4B-Instruct 模型路径；表中 Qwen-VL-4B 名称歧义列为待确认。
- 旧 B0 与新协议参数、judge 和后端不同，禁止复用旧分数到新主表。
- 不在旧环境安装依赖，不运行远端 GPU、不执行真实付费 API、不自动提交/推送 Git。

## Task 1: 来源、数据与提示

**Files:** `evaluation_v2/data.py`, `evaluation_v2/sources.py`, `evaluation_v2/prompts.py`, `scripts/download_eval_v2.py`, `scripts/prepare_eval_v2.py`。

新增七项来源/模型下载和失败隔离，复用已有 SIUO/MSS/Wild 原始数据但不复用回答。所有下载锁定 revision，保存文件 SHA256；按官方字段导入 JSON/CSV/Parquet/ZIP，图片保留字节，统一为可搬运路径。FigStep 使用原始 SafeBench 图像和用户原文 judge prompt；HoliSafe/MMVet/MathVista/WildGuard 提示从锁定源码安全提取字符串，禁止执行第三方脚本。

## Task 2: 严格 HTTP 与可恢复并发执行

**Files:** `evaluation_v2/common.py`, `evaluation_v2/config.py`, `evaluation_v2/transport.py`, `evaluation_v2/runner.py`。

先写网络错误六次尝试、400/401/护栏零重试、无超参数 payload、成功缓存、阶段锁、签名/回答哈希、尾行恢复、鉴权失败停止等测试，再实现。线程池有界提交、benchmark 轮转，单写者 fsync journal；禁止并发写同一阶段，拒答/格式错误不因 resume 再收费。

## Task 3: 八 benchmark 评分和运行质量指标

**Files:** `evaluation_v2/grading.py`, `evaluation_v2/metrics.py`。

MMStar 本地抽取且无 judge；WildJailbreak 仅走 WildGuard completion endpoint；MathVista 抽取与官方类型/精度归一化；MMVet 0–1 score 严格解析。输出每个指标 x/y、完整分母、coverage、护栏率、judge 拒答率、格式错误率、截断数；ASR 只有完整 k 个有效标签入分母，另报缺失取值范围。

## Task 4: 单卡模型切换、计时与并发对照

**Files:** `evaluation_v2/serving.py`, `evaluation_v2/cli.py`, `scripts/evaluate_v2.py`, `scripts/run_eval_v2.sh`, `scripts/sweep_eval_v2.py`。

支持完整模型地址和显式 LoRA adapter，记录配置/权重身份、服务返回模型列表、vLLM/驱动版本。仅清理本次启动的服务进程；单 GPU 两阶段用模型，禁止默认同时加载两个 vLLM 占满显存。保存服务启动时间、每条请求/排队/retry/backoff、每 benchmark span/并集耗时、p50/p95、token/s、整体墙钟和 GPU 采样；sweep 使用同一选择哈希比较并发 1/4/8/16，不宣称随机输出逐字一致或线性多卡加速。

## Task 5: 离线演练和交付

**Files:** `tests/test_eval_v2_*.py`, `scripts/make_eval_v2_demo.py`, `configs/eval_v2*.json`。

本地虚拟 HTTP 服务验证生成/云 judge/WildGuard 三条接口；全部八 benchmark 合成端到端和断点恢复演练，明确标记 mock。运行新定向测试与旧全套测试，确认旧源码指纹不变。编写逐阶段本地下载/打包/服务器预检/smoke/并发对照/全量指令，记录真实数据、GPU、vLLM 未验收项，不预填实测分数。
