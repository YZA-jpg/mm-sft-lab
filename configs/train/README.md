# 后续 SFT 配置

baseline 冻结后在此登记数据配方、训练预算、LoRA 参数和训练环境。当前不提供未经服务器验证的训练入口。

每个配方需要记录训练数据来源/哈希、评测集去污染结果、assistant 监督 token 数、视觉模块冻结策略、训练种子、起始模型和输出 checkpoint；评测复用现有 generation/judge/score 入口。
