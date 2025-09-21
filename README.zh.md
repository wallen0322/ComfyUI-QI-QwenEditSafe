# ComfyUI-QI-QwenEditSafe

中文版本 | [English](README.md)

## 更新日志（中文）

## 新增 / What's New
- **Qwen 2.5 VL 专用加载器（包装） — `QI_QwenVLClipWrapper`**
  - 作用：稳定 VL 视觉输入（几何/分辨率/颜色暗示），提升颜色与一致性稳定性。

- **用法 / Usage** 
1) 工作流中：**官方“加载 CLIP（qwen_image）” → 本节点 → 你的编辑编码器**（`clip` 口对接）。
2) 不改其他节点；仅在本节点按场景切换参数。

- **参数组合 / Quick Recipes**
- **近景人像**：`geometry=center_crop` · `fixed_size=672` · `color_mode=grayscale` · `desaturate=0.15` · `mp_policy=off`
- **全身/半身竖图**：`letterbox` · `896` · `grayscale` · `0.15` · `off`
- **横幅/风景/多人**：`letterbox` · `896` · `grayscale` · `0.15` · `limit`
- **高钥/白底产品**：`letterbox` · `672` · `neutral_gray` · `0.10` · `off`
- **需要改颜色**：按场景选 · `color_mode=original` · `desaturate<=0.05` 或 `neutralize=false`

默认建议 / Default
- `letterbox` · `fixed_size=896` · `color_mode=grayscale` · `desaturate=0.15` · `mp_policy=off`
- 
1) **QI_TextEncodeQwenImageEdit_Safe（图像编辑编码器）**  
- **作用**：将 *提示词+输入图像+VAE* 编码为采样所需的 *conditioning / image / latent*，用于**文字/视觉混合驱动的编辑**。  
- **要点用法**：  
  - `no_resize_pad`：不重采样，仅做信箱式填充，**保持像素一致性**。  
  - `pad_mode` / `grid_multiple`：控制填充方式与网格对齐，减少边缘伪影。  
  - `inject_mode`：控制像素参考注入策略；默认即可。  
  - `encode_fp32`：显存允许时开启，提升稳定性。  
  - `prompt_emphasis`：调“听词”力度（建议 0.4–0.7）。  
  - `vl_max_pixels`：视觉输入上限（1.4MP）自动限流。  
- **连接**：输出的 *conditioning/image/latent* 直接接采样器（KSampler 等）。

2) **QI_RefEditEncode_Safe（一致性编辑编码器）**  
- **作用**：针对**一致性要求高**的编辑（人像/产品/版式等），在保持主体与视感稳定的同时，减少颜色漂移与高光丢层次。  
- **要点用法**：  
  - `out_width/out_height`：**先拉齐输出分辨率**；内部按 32 倍数计算，解码后自动裁回，**像素一一对齐**。  
  - `quality_mode`：`fast / balanced / best / natural` 四挡，平衡速度与质量。  
  - `prompt_emphasis`：只影响**像素参考的收敛强度**，latent 锁脸不变。  
  - 内置 **Linear BT.709 色度对齐 + 亮/暗端衰减**，缓解亮场偏绿/高光顶白；计算上限 3MP，稳定兼容。  
- **连接**：同上，输出直接接采样器；`latent` 内含 `qi_pad` 元信息，供后续解码裁回原设尺寸。

---

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

## 使用说明
安装本节点到comfyui中，可直接下载到custom_nodes目录中，无需额外依赖。
提供两个示例工作流，基础编辑和controlnet应用，
如果您想应用到自己工作流只需要替换TextEncodeQwenImageEdit为本节点就可以，正常使用QI_RefEditEncode_Safe即可，如果想探索更多可能性可以使用QI_TextEncodeQwenImageEdit_Safe自由调节参数。
注意：提示词中需要增加保持人物一致性不变，保持画风光影不变，这句话，效果更佳，

## 特别感谢
- 感谢封号（https://civitai.com/user/Futurlunatic ） ，PT（娃导https://github.com/ptmaster ） 两位大佬的辛苦测试，感谢PT（娃导）的思路，感谢粘土火星（SaturMars：https://github.com/SaturMars ）  提供技术支持，感谢Dontdrunk （https://github.com/Dontdrunk） 的贡献  ，感谢Aiwood研究院小伙伴们的支持。
