# ComfyUI-QI-QwenEditSafe
ComfyUI-QI-QwenEditSafe

# QI • 文生图编辑（中文·一致性增强）

本节点（`QI_TextEncodeQwenImageEdit_CN`）用于在 Qwen Image Edit 的编辑流程中提升**一致性与画质稳定性**，尽量避免在大幅编辑（重构图、改姿态、替换元素）时出现“构图跑偏”“质感被洗”“冷暖色调被改写”等问题。它从**推理阶段**着手，而非后期滤镜：让**条件像素**与 **VAE 参考**同源，支持把 `reference_latents` 与/或 `reference_pixels` 注入到条件里；同时可在 `tokenize(images=[...])` 中引入**辅助原图**与**调色板锚定图**作为多图视觉 token，从源头锁定风格与色域。节点提供**双截棍/FLUX 兼容模式**以降低形状/视图错误和分辨率网格冲突。

## 核心思路
- **同源像素**：用于条件与 VAE 编码的“主对齐图”完全一致，消除缩放/裁切不一致导致的漂移。
- **参考注入**：向 conditioning 写入 `reference_latents` 与/或 `reference_pixels`，把外观锚点直接喂给采样器。
- **多图视觉 token**：除主图外，可加入“轻下采样原图”（巩固身份/光影）与“调色板锚定图”（从原图自动聚类 K 个主色块拼成色板，用于锁定色域与冷暖）。
- **分支兼容**：双截棍/FLUX 模式下自动采用 64 倍网格，并优先像素注入，降低形状/view 报错。

## 适用场景
- 在**必须保持**原图风格、色温与材质的前提下，进行构图或局部大改。
- 人物一致性、品牌色一致性要求高的广告与产品图编辑。
- 需要在 FLUX/双截棍分支下稳定运行的项目。

## 节点输出
- **条件（CONDITIONING）**：内含文本与多图 token 编码结果，以及可选的 `reference_*` 注入。
- **图像（IMAGE）**：对齐后的主图（BHWC，float32，0..1），便于调试或对比。
- **潜空间（LATENT）**：由主图经 `VAE.encode` 产生，供采样器作为 `latent_image`。

## 参数说明
以下为关键参数的中文释义（UI 名称即节点里显示的中文名）：

- **不缩放_保持原尺寸**：尽可能不缩放，仅按网格对齐做裁切；保真度最高，但更吃显存。显存紧张改用“目标总像素”。  
- **目标总像素**：需要缩放时的目标像素数，典型值 1.0–1.5MP，可根据显存上限调高。  
- **网格倍数** 与 **对齐策略**：将分辨率对齐到倍数网格以避免形状错误；标准 Qwen 常用 8/16，双截棍/FLUX 建议 64；“就近对齐”更锐利，“向下对齐”更保守稳定。  
- **保留Alpha**：保留透明通道或强制 RGB3；为稳定起见，常用 RGB3。  
- **下采样插值 / 上采样插值**：缩放插值方式；经验上 **area**（下采样）+ **lanczos**（上采样）画质最稳。  
- **注入模式**：选择把 `reference_latents` 与/或 `reference_pixels` 写入 conditioning；标准 Qwen 推荐“两者都注入”，双截棍/FLUX 推荐“仅像素”或“两者都注入”。  
- **辅助原图参与编码 / 辅助原图像素 / 辅助重复次数**：将轻下采样原图也作为视觉 token 加入编码；有助于身份/光影一致性。非双截棍/FLUX 场景建议开启，重复次数相当于“加权”。  
- **调色板锚定 / 调色板颜色数 / 调色板重复次数**：从原图自动聚类 K 个主色块生成“色板图”，并作为视觉 token 多次重复，直接在编码阶段锁定色域与冷暖；K=5、重复=2 是稳妥起点。  
- **高频轻加强 / 锐化半径 / 锐化强度**：在进入编码之前做极轻微锐化，强调纹理而不改变结构；建议半径 5、强度 0.10–0.15 起步。  
- **兼容双截棍/FLUX**：开启后自动采用 64 倍网格与像素注入偏好，避免该分支常见的形状/view 报错。

## 推荐预设
- **标准 Qwen（锁色 + 保真）**：不缩放=开；网格=8；对齐=就近；注入=两者都注入；辅助原图=开（512×512，重复 2）；调色板锚定=开（K=5，重复 2）；采样建议 steps 22–28 / cfg 3.6–4.2 / denoise 0.25–0.35。  
- **双截棍/FLUX（稳定优先）**：兼容开；不缩放=开（或约 1.2MP）；对齐=向下；注入=仅像素（或两者都注入试验）；辅助原图=关；调色板锚定=关（稳定后可试开）；采样建议 steps 22–26 / cfg 3.6–4.2 / denoise≈0.28。

## 故障排查
- **shape/view 报错**：开启“兼容双截棍/FLUX”或将网格倍数设为 64。  
- **冷色变暖色**：开启“调色板锚定”（K=5，重复 2）；注入模式选“两者都注入”（标准 Qwen）或“仅像素”（双截棍/FLUX）；适度降低 denoise 并加入负面提示“不要改变色温/风格/材质”。  
- **细节发糊**：对齐策略用“就近对齐”，启用高频轻加强（半径 5、强度 0.10–0.15），必要时提高目标总像素。


# QI • TextEncodeQwenImageEdit (Chinese · Consistency-Enhanced)

The node (`QI_TextEncodeQwenImageEdit_CN`) improves **consistency and perceived quality** for Qwen Image Edit. It reduces composition drift, texture washing, and unwanted color-temperature shifts during heavy edits (reframing, pose change, object replacement). The design works at the **inference stage** rather than post-fx: the **conditioning pixels** and the **VAE reference** come from the **same aligned image**, and `reference_latents` and/or `reference_pixels` can be injected into conditioning. You can also feed **an auxiliary downsampled original** and **a palette-anchoring image** as extra visual tokens in `tokenize(images=[...])` to lock style and color gamut. A **Nunchaku/FLUX compatibility** mode minimizes shape/view errors via grid=64 and pixel-preferred injection.

## Core Ideas
- **Same-source pixels** for both conditioning and VAE encode to eliminate drift from inconsistent scaling/cropping.  
- **Reference injection** via `reference_latents` and/or `reference_pixels` to anchor appearance directly for the sampler.  
- **Multi-image tokens**: besides the main aligned image, optionally add a lightly downsampled original (identity/lighting anchor) and an automatically clustered **palette swatch** (K dominant colors) to lock color gamut and temperature from the start.  
- **Fork compatibility**: a dedicated Nunchaku/FLUX mode uses a grid multiple of 64 and prefers pixel injection to avoid shape/view issues.

## When to Use
- Edits that **must preserve** original style, color temperature, and material while changing composition or parts.  
- Identity-critical or brand-color sensitive edits for ads/products.  
- Projects that need robust behavior on FLUX/Nunchaku forks.

## Outputs
- **Conditioning**: text+multi-image token encoding plus optional `reference_*` injection.  
- **Image** (BHWC float32 [0..1]): the aligned main image (handy for inspection).  
- **Latent**: produced by `VAE.encode(main-aligned-image)` to be used as `latent_image` for the sampler.

## Parameters (plain English)
- **No resize; keep original size**: avoid resizing and do grid-aligned cropping only (highest fidelity; more VRAM). Otherwise use **Target total pixels**.  
- **Target total pixels**: desired pixel budget when resizing (1.0–1.5MP typical; increase if VRAM allows).  
- **Grid multiple & snap policy**: snap resolution to a grid to avoid shape errors; 8/16 for standard Qwen, **64 for Nunchaku/FLUX**. “Nearest” is sharper; “Down” is safer.  
- **Keep alpha**: preserve alpha or force RGB3 (RGB3 is usually more stable).  
- **Down/Up resampler**: `area` (down) + `lanczos` (up) are robust defaults.  
- **Injection mode**: choose to inject `reference_latents`, `reference_pixels`, or both. For standard Qwen, **both** is recommended; for Nunchaku/FLUX, choose **pixels** or **both**.  
- **Auxiliary original / pixels / repeats**: also feed a lightly downsampled original as an extra visual token to strengthen identity/lighting (prefer on for standard Qwen; keep off initially for Nunchaku/FLUX). Repeats act as weighting.  
- **Palette anchoring / K colors / repeats**: automatically cluster K dominant colors from the source into a palette swatch and feed it as visual tokens to lock color gamut/temperature. A good starting point is K=5, repeats=2.  
- **High-frequency boost / radius / amount**: a very mild pre-encode unsharp to emphasize texture without altering structure; radius=5 and amount≈0.10–0.15 work well.  
- **Nunchaku/FLUX compatibility**: enables grid=64 and pixel-preferred injection to avoid shape/view issues common to these forks.

## Recommended Presets
- **Standard Qwen (color-locked + fidelity)**: no-resize=on; grid=8; snap=nearest; inject=both; auxiliary image on (512×512, repeats=2); palette anchoring on (K=5, repeats=2). Sampling suggestion: steps 22–28 / cfg 3.6–4.2 / denoise 0.25–0.35.  
- **Nunchaku/FLUX (stability-first)**: compatibility on; no-resize=on (or ~1.2MP target); snap=down; inject=pixels (or both); auxiliary off; palette anchoring off initially. Sampling: steps 22–26 / cfg 3.6–4.2 / denoise≈0.28.

## Troubleshooting
- **Shape/view errors**: enable Nunchaku/FLUX compatibility or set grid multiple to 64.  
- **Cold→Warm shift**: enable palette anchoring (K=5, repeats=2), use **both** (standard Qwen) or **pixels** (Nunchaku/FLUX), slightly reduce denoise, and add negative prompts like “do not change color temperature/style/material.”  
- **Soft details**: use snap=nearest, enable mild high-frequency boost (radius 5, amount 0.10–0.15), and increase target total pixels when possible.

