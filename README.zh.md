# ComfyUI-QI-QwenEditSafe

中文版本 | [English](README.md)

## 简要说明（中文）
- 不缩放网格填充，CLIP 与 VAE 同源输入。
- `reference_pixels` 取自 **VAE 重建**（同域），色彩更稳；`prompt_emphasis` 一个旋钮调"提示词遵从度"。
- 三个节点：文生图编辑（中文）、TextEncodeQwenImageEdit（英文）、VAE Decode。

**快速接线**：`CLIP/图像/VAE` → 文生图编辑 → `(条件, 图像, 潜空间)` → 采样 → VAE Decode。

**常用参数**：
- 注入模式：两者都注入 / 仅 latent / 仅像素（默认 两者都注入）
- 像素来源：VAE 重建（默认）/ 原始图像
- 像素形态：色彩场_64（默认）/ 色彩场_32 / 完整像素
- 文本优先度：0~1（默认 0.5；≥0.8 更听话）
