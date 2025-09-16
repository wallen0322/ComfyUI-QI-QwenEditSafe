# ComfyUI-QI-QwenEditSafe
ComfyUI-QI-QwenEditSafe
QI • 文生图编辑（中文·一致性增强）

ComfyUI 节点：QI_TextEncodeQwenImageEdit_CN（含中文参数与说明）
插件包名：ComfyUI-QI-QwenEditSafe（同时保留英文回退节点 QI_TextEncodeQwenImageEdit_Safe）

🧭 为什么需要它

在使用 Qwen Image Edit 做图像编辑时，常见痛点包括：

一致性漂移：缩放/对齐不一致导致构图、姿态、细节“跑偏”；

色感漂移：冷色输入 → 变成暖色输出（或整体风格被改写）；

分支兼容：FLUX/Nunchaku 等分支对分辨率网格有更严格要求，容易触发 shape 错误。

本节点在推理阶段解决这些问题：

强制 条件像素 ⇄ VAE 参考 同源；

注入 reference_latents / reference_pixels（可选或同时）；

支持 多图视觉 token：可把轻下采样的原图、以及从原图聚类得到的调色板锚定图一并喂给编码器，从根上锁定色感/风格；

自动兼容 FLUX/Nunchaku（64 倍网格），避免 shape/view 报错。

✅ 兼容性

ComfyUI：测试基线 0.3.59

模型：标准 Qwen Image Edit；FLUX/Nunchaku 分支

系统：Windows / Linux（CUDA 环境下测试）

若你使用的是 FLUX/Nunchaku 分支，本节点会默认使用 64 倍网格并优先注入 reference_pixels，以确保稳定。

📦 安装

将插件目录放到：ComfyUI/custom_nodes/ComfyUI-QI-QwenEditSafe/

重启 ComfyUI

在搜索框输入：“文生图编辑（中文·一致性增强）” 即可找到节点

插件同时包含英文回退节点：QI • TextEncodeQwenImageEdit (Safe)（类名 QI_TextEncodeQwenImageEdit_Safe）。

🔌 节点与输出

节点名（中文）：QI • 文生图编辑（中文·一致性增强）
类名：QI_TextEncodeQwenImageEdit_CN
输出：

条件 (CONDITIONING)

图像 (IMAGE, BHWC float32 [0..1]) —— 对齐后的主图（可用于可视化/调试）

潜空间 (LATENT, 形如 {"samples": ...}) —— 由同源像素 VAE.encode 产生
