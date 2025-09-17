# ComfyUI-QI-QwenEditSafe

[中文版本](README.zh.md) | English

## Quick Guide (EN)
- No‑resize grid padding; CLIP & VAE share the same source.
- `reference_pixels` are **VAE recon** (same domain). Use `prompt_emphasis` to trade text adherence vs. reference strength.
- Nodes: TextEncodeQwenImageEdit (EN), 文生图编辑 (CN), VAE Decode.

**Wiring**: `CLIP/IMAGE/VAE` → TextEncodeQwenImageEdit → `(conditioning, image, latent)` → sampler → VAE Decode.

**Key params**:
- inject_mode: both / latents / pixels (default both)
- pixels_source: recon (default) / input
- pixels_shaping: colorfield_64 (default) / colorfield_32 / full
- prompt_emphasis: 0–1 (default 0.5; ≥0.8 = strong text), enable mild high-frequency boost (radius 5, amount 0.10–0.15), and increase target total pixels when possible.
