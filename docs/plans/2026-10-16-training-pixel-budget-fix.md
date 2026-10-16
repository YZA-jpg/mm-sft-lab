# 训练 processor 像素预算兼容修复

问题：服务器 Transformers 4.57.1 的首次 TextVQA 64 条预审在加载 processor 时停止，尚未进行样本编码或训练。原检查只比较 `image_processor.size`，而库的 `from_dict` 可能在构造完成后单独更新 `min_pixels/max_pixels`，使三个字段不同步；fast processor 运行时还会优先使用后两者。

处理：仅在训练入口加载后统一 image processor 的 `size`、`min_pixels`、`max_pixels`，保留显式校验并要求启用 resize。增加 CPU `processor-check`，用低于下限和高于上限的两张合成图片，检查真实 `image_grid_thw` 对应的缩放面积。原始数据、模型目录和评测预处理不修改，训练预算仍为 65,536–1,048,576 像素，总序列仍为 4096。

验证顺序：先加入复现旧字段不同步的离线回归，再修复；检查未启用 resize 和实际缩放越界会被拒绝；运行全套离线测试。真实 Qwen processor 只能在用户同步服务器后验证，随后重跑三组各 64 条预审，仍不启动训练。助手不代用户提交、推送或操作服务器。

依据：[Transformers 4.57.1 加载逻辑](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/image_processing_base.py)、[Qwen fast 图像预处理](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py)。

本地结果：新增回归在修复前复现同一个 ValueError；修复后 7 项 processor 回归和全套 159 项离线测试通过，`git diff --check` 通过。真实服务器 processor 和 token 预审尚待用户同步后验证。
