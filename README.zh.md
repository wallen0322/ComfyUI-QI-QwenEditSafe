# ComfyUI-QI-QwenEditSafe

中文版本 | [English](README.md)

## 简要说明（中文）
- 不缩放网格填充，CLIP 与 VAE 同源输入。
- `reference_pixels` 取自 **VAE 重建**（同域），色彩更稳；`prompt_emphasis` 一个旋钮调"提示词遵从度"。
- 节点：文生图编辑（中文）、**一致性编辑（RefEdit）**、VAE Decode。

**快速接线**：`CLIP/图像/VAE` → 文生图编辑 **或** 一致性编辑（RefEdit） → `(条件, 图像, 潜空间)` → 采样 → VAE Decode。

**常用参数**：
- 质量模式（RefEdit）：natural / fast / balanced / best
- 注入模式：两者都注入 / 仅 latent / 仅像素（默认 两者都注入）
- 像素来源：VAE 重建（默认）/ 原始图像
- 像素形态：色彩场_64（默认）/ 色彩场_32 / 完整像素
- 文本优先度：0~1（默认 0.5；≥0.8 更听话）

## 节点

### QI_TextEncodeQwenImageEdit_Safe
通用文本引导编辑编码器。强调可编辑性，同时通过**不缩放网格填充**与**CLIP/VAE 同源**保持几何稳定。用 **文本优先度** 调节“提示词遵从 ↔ 参考稳定”。

### QI_RefEditEncode_Safe
二段式一致性编辑编码器。前段（0–0.6）重可编辑，后段（0.6–1.0）重一致性与细节；在 ~0.985–1.0 仅薄注入高频，净化边缘/发丝并抑制伪影。提供 **质量模式** 预设（`natural|fast|balanced|best`），默认兼容 Euler/低 CFG。

### QI_VAEDecodeHQ
高质量 VAE 解码。读取 `qi_pad` 元数据自动裁回原尺寸。本身无大改，搭配任一编码器即可。

## 特别感谢
- 感谢封号（https://civitai.com/user/Futurlunatic） ，PT（娃导https://github.com/ptmaster） 两位大佬的辛苦测试，感谢PT（娃导）的思路，感谢粘土火星（SaturMars：https://github.com/SaturMars）  提供技术支持，感谢Aiwood研究院小伙伴们的支持。
