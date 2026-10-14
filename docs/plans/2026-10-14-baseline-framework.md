# Baseline 框架实现计划

目标：为服务器上的 Qwen3-VL-4B-Instruct 提供能执行的 baseline 入口，并支持通过 API 推理和独立 API judge。沿用已有设计，直接修改当前本地仓库；Git 与服务器操作由用户执行。

1. 建立 `src/mm_sft_lab`、严格 JSON 配置及 CLI，保留现有数据准备接口。验证配置错误、prepared 文件及图片哈希、完整配对的小样本选择。
2. 实现本地 Transformers、Chat Completions API 和显式 mock 后端；验证 prompt 不携带标签，批量生成只解码新 token，运行记录与失败恢复不会重复成功项。
3. 实现独立 API judge、MCQA 解析及 SIUO/MSSBench 聚合。验证部分覆盖、三回答联合多数、四类分布和配对分母，不把失败当安全。
4. 加入 preflight、实验比较、离线 demo 与测试，更新服务器启动文档。实际 GPU 和在线 API 验证由用户稍后执行，当前不宣称 baseline 完成。

实现选择：JSON 配置减少初期依赖；API 使用 OpenAI SDK 的 Chat Completions 格式，以接入兼容服务。本地 Qwen 使用官方 processor 默认图片预算、BF16、SDPA、单卡 `cuda:0`；物理卡通过 `CUDA_VISIBLE_DEVICES=7` 指定。

判分采用本项目简明 rubric `lab_rubric_v1`，保留官方标签语义与聚合规则，明确属于适配协议，不宣称逐字复现官方 judge prompt。服务地址与模型由环境变量配置，密钥只从环境变量读取。

验收：离线测试、mock 全链路、CLI help 与语法检查通过；文档提供之后可复制的服务器命令。训练接口留待 baseline 数据与运行协议冻结后实现。

本地完成情况：64 项测试通过；CLI mock 全链路生成 6 条回答、判分 5 条，续跑无重复请求，JSON/CSV/Markdown 产物生成成功。真实 checkpoint 加载、GPU 推理、在线 API 与 judge 配置仍待服务器阶段验证。
