# 本地 judge API 小样本探针

目标：用户先配置本地 DashScope，再用已完成 B0 smoke 的 MSSBench/WildJailbreak 各一条验证真实判分请求与 JSON 解析。原始模型 18/18 成功，3 条截断保留。

设计：新增独立 `scripts/probe_judge.py`，复用既有 judge rubric、ChatClient 和严格标签解析；只支持这两个文本 benchmark，各最多一条，API 重试设为 0，最多两次请求。先验证 run 快照与待测回答，再调用 API；输出独立诊断 JSON，不写正式 judge 目录或评分。配置和服务身份、原始响应、标签、用量与错误均保留，密钥只从环境变量读取且错误脱敏。

实施：编写探针与离线 fake-client 测试，验证两条上限、跳过非语义任务、保存解析错误和拒绝 mock/覆盖输出；更新 API 配置说明及实验记录。助手仅本地编辑/离线测试，用户安装 SDK、配置 Key、SCP 结果和发起真实 API 请求，无需服务器变更。

边界：阿里云当前 DeepSeek 文档仍称仅支持文本输入，SIUO gen 的图片判分后续另行选定视觉服务；本探针不删除 SIUO 图片来绕过限制，也不证明 judge 语义准确率。用户提供的 base_url 先登记为示例，实际使用控制台所属地域/业务空间的 OpenAI 兼容根地址。

来源：[DeepSeek 输入限制与接入地址](https://help.aliyun.com/zh/model-studio/deepseek-api)、[视觉接口示例](https://help.aliyun.com/zh/model-studio/qwen-vl-compatible-with-openai)。

本地验收：7 项 fake-client 测试、CLI help 和 diff 检查通过；真实 API 调用尚未执行。另已核对冻结 commit 的 SIUO 官方 `gpt-eval.py`，safe/effective 两种模式均将原图发送给视觉 judge，现有图片判分协议保留。
