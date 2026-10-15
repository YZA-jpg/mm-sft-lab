# DashScope judge 配置记录

登记日期：2026-10-16。用户提供的配置：

- base_url：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- model：`deepseek-v4.1-flash`
- 调用形式：OpenAI 兼容 Chat Completions；图片使用 Base64 data URL。
- 示例图片路径属于其他项目，本项目使用自己的已锁定 `media_paths`，不会硬编码该路径。
- 密钥未提供；配置模板只记录环境变量名，不包含密钥。

对应 `configs/judge.dashscope.example.json`，使用 `DASHSCOPE_API_KEY`、`JUDGE_BASE_URL`、`JUDGE_MODEL`。仅新增服务示例，没有调用 API、产生判分或变更已有模型输出。

## 验证状态

地址、账号权限、模型 ID、非流式调用及图片输入均待实际连通性检查。阿里云 DeepSeek API 文档列有该模型，但其“图片或文档输入”说明仍称 DeepSeek 仅支持文本；不能据模型原生视觉能力认定同名服务接口一定支持图片。[阿里云 API 文档](https://help.aliyun.com/zh/model-studio/deepseek-api)

SIUO gen 的当前 judge 会发送图片；MSSBench 和 WildJailbreak 的当前 rubric 只发送文本。正式调用 `judge` 会覆盖该 run 的全部语义任务，目前不支持在同一 judge run 中自动按任务切换不同服务模型。图片验证失败时，应先确定视觉 judge 或另行实现任务路由，不能删掉 SIUO 图片继续冒充同一协议。

模板显式设置 `enable_thinking=false`，以 512 输出 token 生成简短 JSON；该设置也必须在连通性测试中确认平台接受。修改思考模式、模型或 rubric 应采用新 judge 目录并冻结后再统一判四个模型。

## 调用位置与后续步骤

优先在本地调用 API。服务器完成生成后，把完整 run 目录传回本地；运行 judge 时显式指定本地 `--data-root data/step1`，使用与服务器相同 manifest 的数据快照。原始服务器路径不会直接用于本地图片读取。

连通性测试作为独立阶段进行：先配置本地 API 依赖和环境变量；再分别验证一条纯文本和一条无风险图片输入；最后验证 rubric JSON 与人工边界样本。数据传输阶段不接着批量 judge。

本地 PowerShell 环境变量示例（等 judge 连通性阶段再执行）：

```powershell
$env:JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:JUDGE_MODEL = "deepseek-v4.1-flash"
$env:DASHSCOPE_API_KEY = [System.Net.NetworkCredential]::new(
    "", (Read-Host "输入 DashScope API Key" -AsSecureString)
).Password
```

密钥只进入当前进程环境；脚本不会自动读取 `.env`。不要把实际密钥写进 Git 配置、实验 manifest 或提交记录。
