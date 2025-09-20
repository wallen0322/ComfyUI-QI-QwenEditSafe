from __future__ import annotations
import torch
import torch.nn.functional as F
import node_helpers, comfy.utils

_ALIGN_MULTIPLE = 32
_SAFE_MAX_PIXELS = 3000000  # 3MP hard cap

def _ceil_to(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m if m > 1 else v

def _to_bhwc_any(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        x = x[0]
    if x.ndim == 3:
        if x.shape[0] in (1, 3, 4):
            x = x.unsqueeze(0).movedim(1, -1)
        elif x.shape[-1] in (1, 3, 4):
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(0).movedim(1, -1)
    elif x.ndim == 4:
        if x.shape[-1] in (1, 3, 4):
            pass
        elif x.shape[1] in (1, 3, 4):
            x = x.movedim(1, -1)
        else:
            x = x.movedim(1, -1)
    return x.float().clamp(0, 1).contiguous()

def _ensure_rgb3(bhwc: torch.Tensor) -> torch.Tensor:
    c = int(bhwc.shape[-1])
    if c == 3:
        return bhwc
    if c == 1:
        return bhwc.repeat(1, 1, 1, 3)
    return bhwc[..., :3]

def _bhwc(x: torch.Tensor) -> torch.Tensor:
    return _ensure_rgb3(_to_bhwc_any(x))

def _bchw(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-1, 1).contiguous()

def _choose_method(sw: int, sh: int, tw: int, th: int) -> str:
    if tw * th <= 0:
        return "area"
    if tw >= sw or th >= sh:
        return "lanczos"
    return "area"

def _resize_bchw(x: torch.Tensor, w: int, h: int, method: str) -> torch.Tensor:
    return comfy.utils.common_upscale(x, w, h, method, "disabled")

def _resize_bchw_smart(x: torch.Tensor, w: int, h: int) -> torch.Tensor:
    _, _, H, W = x.shape
    return _resize_bchw(x, w, h, _choose_method(W, H, w, h))

def _fit_letterbox(src_bhwc: torch.Tensor, Wt: int, Ht: int, pad_mode: str = "reflect") -> tuple[torch.Tensor, dict]:
    B, H, W, C = src_bhwc.shape
    if W == Wt and H == Ht:
        return src_bhwc, {"top": 0, "bottom": 0, "left": 0, "right": 0}
    sx = Wt / float(W)
    sy = Ht / float(H)
    s = min(sx, sy)
    Wr = max(1, int(round(W * s)))
    Hr = max(1, int(round(H * s)))
    base = _ensure_rgb3(_resize_bchw_smart(_bchw(src_bhwc), Wr, Hr).movedim(1, -1))
    top = (Ht - Hr) // 2
    bottom = Ht - Hr - top
    left = (Wt - Wr) // 2
    right = Wt - Wr - left
    bchw = _bchw(base)
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        bchw = F.pad(bchw, (left, right, top, bottom), mode=pad_mode)
    return _ensure_rgb3(bchw.movedim(1, -1)), {"top": int(top), "bottom": int(bottom), "left": int(left), "right": int(right)}

def _apply_vl_cap(img_bhwc: torch.Tensor) -> torch.Tensor:
    cap = 1_400_000  # single upper bound (1.4MP)
    Ht, Wt = int(img_bhwc.shape[1]), int(img_bhwc.shape[2])
    area = Ht * Wt
    if area <= cap:
        return img_bhwc
    scale = (cap / float(area)) ** 0.5
    Hs = max(1, int(Ht * scale))
    Ws = max(1, int(Wt * scale))
    return _ensure_rgb3(_resize_bchw_smart(_bchw(img_bhwc), Ws, Hs).movedim(1, -1))

def _color_lock_to(x_bhwc: torch.Tensor, ref_bhwc: torch.Tensor, mix: float = 0.97) -> torch.Tensor:
    x = _bchw(_ensure_rgb3(x_bhwc))
    r = _bchw(_ensure_rgb3(ref_bhwc))
    R, G, B = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    Yx = 0.299 * R + 0.587 * G + 0.114 * B
    Cbx = (B - Yx) * 0.564
    Crx = (R - Yx) * 0.713
    Rr, Gr, Br = r[:, 0:1], r[:, 1:2], r[:, 2:3]
    Yr = 0.299 * Rr + 0.587 * Gr + 0.114 * Br
    Cbr = (Br - Yr) * 0.564
    Crr = (Rr - Yr) * 0.713
    cbx_mu = Cbx.mean(dim=(2, 3), keepdim=True)
    cbx_std = Cbx.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    crx_mu = Crx.mean(dim=(2, 3), keepdim=True)
    crx_std = Crx.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    cbr_mu = Cbr.mean(dim=(2, 3), keepdim=True)
    cbr_std = Cbr.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    crr_mu = Crr.mean(dim=(2, 3), keepdim=True)
    crr_std = Crr.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    cb_scale = (cbr_std / cbx_std).clamp(0.99, 1.01)
    cr_scale = (crr_std / crx_std).clamp(0.99, 1.01)
    cb_shift = (cbr_mu - cbx_mu).clamp(-0.004, 0.004)
    cr_shift = (crr_mu - crx_mu).clamp(-0.004, 0.004)
    Cb_aligned = (Cbx - cbx_mu) * cb_scale + cbx_mu + cb_shift
    Cr_aligned = (Crx - crx_mu) * cr_scale + crx_mu + cr_shift
    Y_aligned = Yx
    R2 = (Y_aligned + 1.403 * Cr_aligned).clamp(0, 1)
    G2 = (Y_aligned - 0.344 * Cb_aligned - 0.714 * Cr_aligned).clamp(0, 1)
    B2 = (Y_aligned + 1.773 * Cb_aligned).clamp(0, 1)
    aligned = torch.cat([R2, G2, B2], dim=1)
    y = (mix * aligned + (1.0 - mix) * x).movedim(1, -1)
    return _ensure_rgb3(y)

def _lowpass_ref(bhwc: torch.Tensor, size: int = 64) -> torch.Tensor:
    bchw = _bchw(bhwc)
    bh = F.interpolate(bchw, size=(size, size), mode="area")
    bh = _resize_bchw_smart(bh, bchw.shape[-1], bchw.shape[-2])
    return _ensure_rgb3(bh.movedim(1, -1)).clamp(0, 1)

def _hf_ref_smart(bhwc: torch.Tensor, alpha: float, blur_k: int, kstd: float, bias: float, smooth_k: int) -> torch.Tensor:
    bchw = _bchw(bhwc)
    k = max(3, blur_k | 1)
    pad = k // 2
    if bchw.shape[2] <= pad or bchw.shape[3] <= pad:
        return _ensure_rgb3(bchw.movedim(1, -1))
    base = F.avg_pool2d(bchw, kernel_size=k, stride=1, padding=pad)
    detail = bchw - base
    ker = torch.ones((1, 1, 3, 3), dtype=bchw.dtype, device=bchw.device) / 9.0
    y = (_bchw(bhwc)[:, 0:1] * 0.299 + _bchw(bhwc)[:, 1:2] * 0.587 + _bchw(bhwc)[:, 2:3] * 0.114)
    mu = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), ker)
    var = F.conv2d(F.pad((y - mu) ** 2, (1, 1, 1, 1), mode="replicate"), ker)
    std = torch.sqrt(var + 1e-6)
    t = kstd * std + bias
    absd = torch.abs(detail)
    t3 = torch.cat([t, t, t], dim=1)
    d = torch.sign(detail) * torch.clamp(absd - t3, min=0.0)
    e = torch.sqrt(
        F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"),
                 torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 8.0) ** 2 +
        F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"),
                 torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 8.0) ** 2
    ) + 1e-6
    mid = torch.exp(-((y - 0.5) ** 2) / (2 * 0.25 * 0.25))
    tex = (std / (std.mean(dim=(2, 3), keepdim=True) + 1e-6)).clamp(0, 1.5)
    tex = (tex - 0.2) / 0.8
    tex = tex.clamp(0, 1)
    gate = (0.16 + 0.40 * (e / (e.mean(dim=(2, 3), keepdim=True) * 3.0)) + 0.10 * tex) * (0.85 + 0.15 * mid)
    gate3 = torch.cat([gate, gate, gate], dim=1)
    dm = torch.tanh(d * 2.0) * 0.38
    out = (bchw + alpha * (dm * gate3)).clamp(0, 1)
    return _ensure_rgb3(out.movedim(1, -1))

class QI_TextEncodeQwenImageEdit_Safe:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("CONDITIONING", "IMAGE", "LATENT")
    RETURN_NAMES = ("conditioning", "image", "latent")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "image": ("IMAGE",),
            "vae": ("VAE",),
            "out_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
            "out_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
            "prompt_emphasis": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    def encode(self, clip, prompt, image, vae, out_width=0, out_height=0, prompt_emphasis=0.60):
        src = _bhwc(image)[..., :3]
        H, W = int(src.shape[1]), int(src.shape[2])
        Wtgt = int(out_width) if int(out_width) > 0 else W
        Htgt = int(out_height) if int(out_height) > 0 else H

        area_tgt = Wtgt * Htgt
        if area_tgt > _SAFE_MAX_PIXELS:
            s = (_SAFE_MAX_PIXELS / float(area_tgt)) ** 0.5
            Wtgt = max(8, int(Wtgt * s))
            Htgt = max(8, int(Htgt * s))
        Wcmp = _ceil_to(Wtgt, _ALIGN_MULTIPLE)
        Hcmp = _ceil_to(Htgt, _ALIGN_MULTIPLE)

        letter, pad_ext = _fit_letterbox(src, Wtgt, Htgt, pad_mode="reflect")
        top = pad_ext["top"]; left = pad_ext["left"]; bottom = pad_ext["bottom"]; right = pad_ext["right"]
        ph, pw = Hcmp - Htgt, Wcmp - Wtgt
        if ph > 0 or pw > 0:
            top2 = ph // 2; bottom2 = ph - top2; left2 = pw // 2; right2 = pw - left2
            bchw = _bchw(letter)
            bchw = F.pad(bchw, (left2, right2, top2, bottom2), mode="reflect")
            letter = _ensure_rgb3(bchw.movedim(1, -1))
            top += top2; bottom += bottom2; left += left2; right += right2

        padded = letter.contiguous()
        vl_img = _apply_vl_cap(padded)
        tokens = clip.tokenize(prompt, images=[vl_img])
        cond = clip.encode_from_tokens_scheduled(tokens)

        with torch.inference_mode():
            lat = vae.encode(padded)
            if isinstance(lat, dict) and "samples" in lat:
                lat = lat["samples"]
            if isinstance(lat, torch.Tensor) and lat.dtype != torch.float32:
                lat = lat.float()

        emph = float(max(0.0, min(1.0, prompt_emphasis)))
        pixM = padded
        pixE = _lowpass_ref(padded, 64)
        pixL = _hf_ref_smart(padded, alpha=0.16, blur_k=3, kstd=0.88, bias=0.0045, smooth_k=5)

        ref_scale = max(0.70, min(1.05, 1.05 - 0.35 * emph))
        E_w = max(0.30, min(0.80, 0.35 + 0.30 * emph)) * ref_scale
        M_w = 0.50 * (1.0 - 0.4 * (emph - 0.5)) * ref_scale
        L_w = max(0.15, min(0.45, 0.18 + 0.20 * emph)) * ref_scale
        lat_w = 1.0 * (1.0 + 0.6 * (emph - 0.5))

        pixM = _color_lock_to(pixM, padded, mix=0.97)
        pixE = _color_lock_to(pixE, padded, mix=0.97)
        pixL = _color_lock_to(pixL, padded, mix=0.97)

        cond = node_helpers.conditioning_set_values(cond, {"reference_latents": [lat], "strength": float(lat_w), "timestep_percent_range": [0.0, 0.40 if emph < 0.7 else 0.36]}, append=True)
        cond = node_helpers.conditioning_set_values(cond, {"reference_pixels": [pixE], "strength": float(E_w), "timestep_percent_range": [0.30, 0.58]}, append=True)
        cond = node_helpers.conditioning_set_values(cond, {"reference_pixels": [pixM], "strength": float(M_w), "timestep_percent_range": [0.45, 0.75]}, append=True)
        cond = node_helpers.conditioning_set_values(cond, {"reference_pixels": [pixL], "strength": float(L_w), "timestep_percent_range": [0.82, 1.00]}, append=True)

        latent = {"samples": lat, "qi_pad": {"top": int(top), "bottom": int(bottom), "left": int(left), "right": int(right),
                                             "orig_h": int(Htgt), "orig_w": int(Wtgt),
                                             "compute_h": int(padded.shape[1]), "compute_w": int(padded.shape[2])}}
        return (cond, _ensure_rgb3(_bhwc(image)), latent)

NODE_CLASS_MAPPINGS = {"QI_TextEncodeQwenImageEdit_Safe": QI_TextEncodeQwenImageEdit_Safe}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_TextEncodeQwenImageEdit_Safe": "Qwen图像编辑编码器 — by wallen0322"}
