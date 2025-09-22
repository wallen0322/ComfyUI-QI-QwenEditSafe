# ComfyUI-QI-QwenEditSafe

[中文版本](README.zh.md) | English

**Updated Nodes**

- **New Node:** *Qwen 2.5 VL Wrapper* — `QI_QwenVLClipWrapper`  
  Purpose: stabilize VL preprocessing (geometry/resize/color hints) for better color & consistency.
- **Hook:** Official *Load CLIP (qwen_image)* → **Wrapper** → your encoder (`clip`).
- **Recipes:** Portrait `center_crop·672·grayscale·0.15·off`; Full/vertical `letterbox·896·grayscale·0.15·off`; Landscape `letterbox·896·grayscale·0.15·limit`; High‑key `letterbox·672·neutral_gray·0.10·off`; Color‑change `color_mode=original` + low desat (≤0.05) or neutralize=false.
- **Default:** `letterbox·896·grayscale·0.15·off`


1) **QI_TextEncodeQwenImageEdit_Safe (Image Edit Encoder)**  
- **Purpose**: Encode *prompt + input image + VAE* into *conditioning / image / latent* for text–vision mixed editing.  
- **How to use**:  
  - `no_resize_pad`: letterbox-only (no resample) to **preserve pixel consistency**.  
  - `pad_mode` / `grid_multiple`: control padding and grid alignment to reduce edge artifacts.  
  - `inject_mode`: pixel-reference injection strategy (defaults are fine).  
  - `encode_fp32`: enable if VRAM allows for stability.  
  - `prompt_emphasis`: strength of prompt adherence (suggest 0.4–0.7).  
  - `vl_max_pixels`: CLIP-vision cap (1.4MP) auto-throttled.  
- **Wiring**: feed outputs directly into your sampler (e.g., KSampler).

2) **QI_RefEditEncode_Safe (Consistency Edit Encoder)**  
- **Purpose**: High-consistency editing (portraits, products, layout) with reduced color drift and preserved highlight/shadow detail.  
- **How to use**:  
  - `out_width/out_height`: **lock output size first**; the pipeline computes at 32× alignment and crops back, giving **pixel-perfect alignment**.  
  - `quality_mode`: choose among `fast / balanced / best / natural` for speed/quality trade-off.  
  - `prompt_emphasis`: affects **only pixel-reference convergence**; latent lock remains.  
  - Built-in **Linear BT.709 chroma match + highlight/lowlight attenuation** to mitigate bright-scene green shift & highlight clipping; 3MP compute cap for stability.  
- **Wiring**: same—pipe *conditioning/image/latent* to the sampler; `latent.qi_pad` carries crop metadata for final decoding.


## Quick Overview (EN)
- No-resize grid padding; CLIP and VAE share the same source image.
- `reference_pixels` come from **VAE reconstruction** (same domain) for more stable colors; `prompt_emphasis` is a single knob for text adherence.
- Nodes: CN Text Edit, **Consistency Edit (RefEdit)**, VAE Decode.

**Quick Wiring**: `CLIP / IMAGE / VAE` → CN Text Edit **or** Consistency Edit (RefEdit) → `(conditioning, image, latent)` → sampler → VAE Decode.

**Common Parameters**:
- Quality mode (RefEdit): natural / fast / balanced / best
- Injection mode: both / latent only / pixels only (default: both)
- Pixel source: VAE reconstruction (default) / original image
- Pixel form: color_field_64 (default) / color_field_32 / full pixels
- Prompt emphasis: 0–1 (default 0.5; ≥0.8 follows text more)

## Nodes

### QI_TextEncodeQwenImageEdit_Safe
General text-guided edit encoder. Emphasizes editability while keeping geometry stable via **no-resize grid padding** and **shared CLIP/VAE source**. Use **prompt emphasis** to balance “text adherence ↔ reference stability.”

### QI_RefEditEncode_Safe
Two-phase consistency edit encoder. Early (0–0.6) focuses on editability; Late (0.6–1.0) reinforces consistency and detail. A thin high-frequency tail around ~0.985–1.0 cleans edges/hair and suppresses artifacts. Provides **quality mode** presets (`natural|fast|balanced|best`) and is Euler / low-CFG friendly by default.

### QI_VAEDecodeHQ
High-quality VAE decode. Reads `qi_pad` metadata and auto-crops back to the original size. No major changes; pair with either encoder.

## Usage
Install this package into ComfyUI (place under `custom_nodes`), no extra dependencies required.  
Two sample workflows are provided: basic editing and ControlNet usage.  
If you want to use it in your own workflow, just replace **TextEncodeQwenImageEdit** with this node and use **QI_RefEditEncode_Safe** in normal cases; for more exploration, use **QI_TextEncodeQwenImageEdit_Safe** and adjust parameters freely.  
Note: Add the sentence “保持人物一致性不变，保持画风光影不变” to your prompt for better results.

## Special Thanks
- Thanks to 封号 (https://civitai.com/user/Futurlunatic) and PT（娃导 https://github.com/ptmaster ） for rigorous testing; thanks to PT（娃导） for the idea; thanks to 粘土火星（SaturMars：https://github.com/SaturMars ） for technical support; and thanks to the Aiwood 研究院 teammates for their support.
