# mm-sft-lab

研究多模态 SFT 数据对模型安全性与能力的影响。当前先完成 Qwen3-VL-4B-Instruct baseline，训练阶段暂缓。

当前操作入口只有 [baseline 执行手册](docs/baseline-runbook.md)。文档导航、历史记录和训练资料集中在 [文档索引](docs/README.md)，数据处理与协议变更保存在 [实验记录](docs/data-and-experiment-notes.md)。

## 当前运行方案（2026-10-23）

| 项目 | 设置 |
| --- | --- |
| 本轮已准备的数据 | SIUO gen、MSSBench chat / embodied |
| 次数 | SIUO 每题 3 次；MSSBench 每条 1 次，保留完整配对 |
| 目标模型 | max_tokens=65536；其他解码参数沿用服务默认值 |
| 调度 | 数据集间串行；单数据集内最多 4 个并发请求 |
| 验证顺序 | 全输入离线预检 → 新 15 条 smoke → 人工检查反馈 → 单独启动新全量 |
| 全量规模 | 501 + 1960 = 2461 次生成 |
| 本轮启动范围 | 仅生成；判分单独进行，不在 smoke 中调用付费 API |

WebP MIME 不再依赖操作系统注册表；启动前构造所有选中请求。新 smoke 覆盖全部 5 张 WebP，包括此前失败的样本及其配对。旧运行保留为诊断记录，新结果不回填到旧目录。

八项最终 benchmark 清单及适配差异见 [来源说明](docs/evaluation-v2-sources.md)；其他六项尚未与本轮 SIUO/MSSBench 一同完成服务器验收。

## 本地检查

```bash
python -m unittest discover -s tests -p 'test_eval_v2_*.py'
```

助手负责本地修改和离线验证；Git 发布、服务器安装、GPU 推理和真实 API 调用由用户执行。具体约定见 [工作流](docs/workflow.md)。数据、模型、密钥和运行产物不提交 Git。
