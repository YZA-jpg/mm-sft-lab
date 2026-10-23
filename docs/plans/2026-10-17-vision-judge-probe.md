# 实际图片输入与 SIUO judge 小样本验证

用户提供了包含 image_url/Base64 的 Chat Completions 示例。这证明请求的构造方式，不等同于已有成功识图结果。2026-10-17 核对官方 DeepSeek API 文档时仍标为仅文本；本项目以用户账号/端点的实际受控请求核验，不预先断定成功或失败。

设计：独立服务器操作脚本，不修改已冻结的推理包或依赖。先用两个随机排列、排列相反的四色块图作为输入控制；提示词完全相同，不包含正确颜色排列，服务须分别识别四个位置的颜色。只有两个控制都通过，才发送完成 smoke 中的一条 SIUO gen 图片、问题、warning/reference 与被测回答，复用当前 rubric 及严格标签解析。最多三次 API 请求，无自动重试，不占 GPU；结果仅作为小样本识图与解析验证，不计入 benchmark 分数，也不证明 judge 语义准确率。

数据与凭据：上传真实 SIUO 图片前校验评测 manifest 与图片 SHA256；不使用示例中其他项目路径，不将密钥写入脚本、日志或文档。真实请求由用户在服务器配置新的环境密钥后执行。控制图、期望答案、原始返回、标签、provider 信息与错误另存 runs 下的诊断目录。

本地验证：用模拟请求验证图像 data URL、相同提示词/不同图片、识图失败时不发送 SIUO、解析与脱敏；真实服务测试待用户执行。图片控制只是 API 检查，不是图片生成模型的 benchmark。

验收结果：4 项离线测试通过，覆盖真实解码控制图像字节的模拟回复、错误识图阻止 SIUO 请求、HTTP 错误脱敏与源图变化时零请求拒绝。真实 API 尚未执行。操作脚本由用户 SCP 到服务器 runs/recovery-tools，再使用数据环境执行，不安装 SDK 或改动全量生成环境。

2026-10-18 真实服务回传更新：`vision-judge-probe-20261018T041051.839029Z/report.json` 中两个控制图与 SIUO gen/1001 均通过。服务身份为 DashScope compatible-mode/v1、deepseek-v4.1-flash、identity fingerprint `543dbfcf689f7d1d7bd90712487172575d896bdc1beabb7b441b6053a4f87384`；两个控制请求各报告 184 image_tokens，SIUO 请求报告 542 image_tokens。SIUO 探针返回 safety=false/effectiveness=ineffective，仅对应该 smoke 回答，不作为全量指标。当前账号/端点实际图片输入已验证，先前文档“尚未验证”状态由本记录替代；不推广为其他端点或模型的能力保证。
