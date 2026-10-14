# Smoke 与小批量测试实现

目标：在本地准备好可由用户稍后在服务器执行的测试代码，不启动真实模型、不修改现有 ipiguard 环境。

1. 固定三份配置：12 条 smoke / batch 1，同样 12 条 / batch 4，48 条小规模 / batch 4。验证两个 smoke 清单完全一致、小规模清单包含完整 MSSBench 配对。
2. 在生成记录中保存实际 batch ID/大小/耗时；本地 Qwen 记录视觉 grid、padding 宽度与 PyTorch 显存峰值。验证 padded token 裁切与视觉输入数量，记录 OOM 拆批次数。
3. 增加 `validate-run` 与 `compare-batches`：独立检查覆盖率、空回答、截断、MCQA 格式、视觉输入及批量输出差异；输出 JSON、Markdown 和逐条复核文件。测试失败、清单不一致、协议变化、全部拆成单条等情况不能通过。
4. 增加服务器分阶段包装脚本与操作文档；用离线 fake/mock 后端验证完整流程，用真实 prepared 数据验证 12/12/48 选样。GPU 与真实 CUDA/API 验证留给用户回来后执行。

比较允许变化的生成参数仅为 batch_size。输出文本的差异需要人工复核，不直接认定为模型或代码错误。mock 结果明确标记，不能使真实 baseline 就绪检查通过。

本地完成验证：83 项离线测试通过；Bash 语法与包装脚本的 plan 阶段通过；真实 prepared 数据选样为 12 / 12 / 48，前两者哈希一致。mock CLI 的三阶段 generate / validate-run 与 compare-batches 通过，全部标记模拟结果；未调用真实模型或 judge。操作入口见 [smoke 与小批量文档](../smoke-tests.md)。
