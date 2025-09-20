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

def _smoothstep(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - a) / max(1e-6, (b - a)))
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.0031308, t * 12.92, 1.055 * torch.pow(t, 1.0 / 2.4) - 0.055)

def _batched_cov_2x2(Cb: torch.Tensor, Cr: torch.Tensor, mask: torch.Tensor):
    # Cb/Cr/Mask: Bx1xHxW (float)
    eps = 1e-6
    B, _, H, W = Cb.shape
    m = mask.float()
    denom = m.sum(dim=(2,3), keepdim=True).clamp_min(1.0)
    mu_cb = (Cb * m).sum(dim=(2,3), keepdim=True) / denom
    mu_cr = (Cr * m).sum(dim=(2,3), keepdim=True) / denom
    dc_b = (Cb - mu_cb) * m
    dc_r = (Cr - mu_cr) * m
    var_cb = (dc_b * (Cb - mu_cb)).sum(dim=(2,3), keepdim=True) / denom + eps
    var_cr = (dc_r * (Cr - mu_cr)).sum(dim=(2,3), keepdim=True) / denom + eps
    cov_br = (dc_b * (Cr - mu_cr)).sum(dim=(2,3), keepdim=True) / denom
    # Pack Bx2x2
    C = torch.zeros((B, 2, 2), dtype=Cb.dtype, device=Cb.device)
    C[:,0,0] = var_cb.squeeze()
    C[:,1,1] = var_cr.squeeze()
    C[:,0,1] = cov_br.squeeze()
    C[:,1,0] = cov_br.squeeze()
    mu = torch.cat([mu_cb, mu_cr], dim=1)  # Bx2x1x1
    return mu, C

def _sym_sqrtm_2x2(C: torch.Tensor):
    # C: Bx2x2 SPD
    eps = 1e-6
    w, V = torch.linalg.eigh(C)  # Bx2, Bx2x2
    w = w.clamp_min(eps)
    sqrtw = torch.sqrt(w)
    invsqrtw = 1.0 / sqrtw
    S = V @ torch.diag_embed(sqrtw) @ V.transpose(-1, -2)
    Sinv = V @ torch.diag_embed(invsqrtw) @ V.transpose(-1, -2)
    return S, Sinv

def _apply_chroma_whiten_color(Cb_x, Cr_x, Cb_r, Cr_r, Y_lin):
    # Inputs: Bx1xHxW tensors (linear domain)
    # Build midtone mask for robust stats (avoid deep shadow / near-white)
    B, _, H, W = Y_lin.shape
    yvec = Y_lin.reshape(B, 1, H*W)
    lo = torch.quantile(yvec, 0.06, dim=-1, keepdim=True).reshape(B,1,1,1)
    hi = torch.quantile(yvec, 0.94, dim=-1, keepdim=True).reshape(B,1,1,1)
    mask = (Y_lin >= lo) & (Y_lin <= hi)

    mu_x, Cx = _batched_cov_2x2(Cb_x, Cr_x, mask)
    mu_r, Cr = _batched_cov_2x2(Cb_r, Cr_r, mask)

    S_r, _ = _sym_sqrtm_2x2(Cr)
    _, Sinv_x = _sym_sqrtm_2x2(Cx)
    # T: Bx2x2
    T = S_r @ Sinv_x

    # Clamp singular values of T to near-identity (±3%)
    U, S, Vh = torch.linalg.svd(T)
    S = S.clamp(0.97, 1.03)
    T = U @ torch.diag_embed(S) @ Vh

    # Shift clamp ±0.004
    dmu = (mu_r - mu_x)  # Bx2x1x1
    dmu = dmu.clamp(-0.004, 0.004)

    # Apply to each pixel: v' = T (v - mu_x) + (mu_x + dmu)
    B,_,H,W = Cb_x.shape
    v = torch.cat([Cb_x, Cr_x], dim=1).reshape(B, 2, H*W)        # Bx2xN
    mu_x2 = mu_x.reshape(B, 2, 1)
    dmu2 = dmu.reshape(B, 2, 1)
    vp = (T @ (v - mu_x2)) + (mu_x2 + dmu2)                     # Bx2xN
    Cb_p = vp[:,0].reshape(B,1,H,W)
    Cr_p = vp[:,1].reshape(B,1,H,W)
    return Cb_p, Cr_p

def _color_lock_to(x_bhwc: torch.Tensor, ref_bhwc: torch.Tensor, mix: float = 0.97) -> torch.Tensor:
    # 1) To linear domain
    x_srgb = _bchw(_ensure_rgb3(x_bhwc))
    r_srgb = _bchw(_ensure_rgb3(ref_bhwc))
    x_lin = _srgb_to_linear(x_srgb)
    r_lin = _srgb_to_linear(r_srgb)

    R, G, B = x_lin[:,0:1], x_lin[:,1:2], x_lin[:,2:3]
    Rr, Gr, Br = r_lin[:,0:1], r_lin[:,1:2], r_lin[:,2:3]

    # 2) BT.709 luminance (linear)
    Yx = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Yr = 0.2126 * Rr + 0.7152 * Gr + 0.0722 * Br

    # BT.709 chroma scales
    cb_s = 0.5 / (1.0 - 0.0722)
    cr_s = 0.5 / (1.0 - 0.2126)

    Cbx = (B - Yx) * cb_s
    Crx = (R - Yx) * cr_s
    Cbr = (Br - Yr) * cb_s
    Crr = (Rr - Yr) * cr_s

    # 3) Chroma match via whiten->color (tiny-clamped) on midtones
    Cb_aligned, Cr_aligned = _apply_chroma_whiten_color(Cbx, Crx, Cbr, Crr, Yx)

    # 4) Luminance-preserving reconstruction (linear)
    inv_cb = 1.0 / cb_s
    inv_cr = 1.0 / cr_s
    R2 = (Yx + Cr_aligned * inv_cr)
    B2 = (Yx + Cb_aligned * inv_cb)
    G2 = (Yx - 0.2126 * R2 - 0.0722 * B2) / 0.7152
    aligned_lin = torch.cat([R2, G2, B2], dim=1).clamp(0, 1)

    # 5) Highlight/shadow attenuation for the blend
    w_hi = 1.0 - _smoothstep(0.75, 0.98, Yx)
    w_lo = _smoothstep(0.05, 0.12, Yx)
    w = (mix * w_hi * w_lo).clamp(0.0, 1.0)

    out_lin = w * aligned_lin + (1.0 - w) * x_lin
    out_srgb = _linear_to_srgb(out_lin).movedim(1, -1)
    return _ensure_rgb3(out_srgb)

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

class QI_RefEditEncode_Safe:
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
            "prompt_emphasis": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            "quality_mode": (["natural", "fast", "balanced", "best"], {"default": "natural"}),
        }}

    def encode(self, clip, prompt, image, vae, out_width=0, out_height=0, prompt_emphasis=0.6, quality_mode="natural"):
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
        if quality_mode == "fast":
            lat_e, pixM_e, pixE_e = 0.68, 0.42, 0.20; lat_l, pixM_l, pixL_l = 0.95, 0.78, 0.02; hf_alpha, kstd, bias, smooth = 0.14, 0.84, 0.0045, 5
        elif quality_mode == "best":
            lat_e, pixM_e, pixE_e = 0.76, 0.52, 0.26; lat_l, pixM_l, pixL_l = 1.05, 0.90, 0.03; hf_alpha, kstd, bias, smooth = 0.17, 0.90, 0.0045, 7
        elif quality_mode == "balanced":
            lat_e, pixM_e, pixE_e = 0.72, 0.48, 0.24; lat_l, pixM_l, pixL_l = 1.00, 0.85, 0.025; hf_alpha, kstd, bias, smooth = 0.16, 0.88, 0.0045, 5
        else:
            lat_e, pixM_e, pixE_e = 0.73, 0.49, 0.24; lat_l, pixM_l, pixL_l = 1.02, 0.86, 0.025; hf_alpha, kstd, bias, smooth = 0.16, 0.88, 0.0045, 5

        ref_scale = max(0.70, min(1.05, 1.05 - 0.35 * emph))
        pixM_e *= ref_scale; pixM_l *= ref_scale; pixE_e *= ref_scale

        pixM = padded if (pixM_e > 0 or pixM_l > 0) else None
        pixE = _lowpass_ref(padded, 64) if pixE_e > 0 else None
        pixL = _hf_ref_smart(padded, hf_alpha, 3, kstd, 0.0045, smooth) if pixL_l > 0 else None

        if pixM is not None: pixM = _color_lock_to(pixM, padded, mix=0.97)
        if pixE is not None: pixE = _color_lock_to(pixE, padded, mix=0.97)
        if pixL is not None: pixL = _color_lock_to(pixL, padded, mix=0.97)

        early = (0.0, 0.6); late = (0.6, 1.0); hf_rng = (0.985, 1.0)

        def _add_ref(c, rng, lat_w, pixE_w, pixM_w, pixL_w, hfr=None):
            c = node_helpers.conditioning_set_values(c, {"reference_latents": [lat], "strength": float(lat_w), "timestep_percent_range": [float(rng[0]), float(rng[1])]}, append=True)
            if pixE is not None and pixE_w > 0:
                c = node_helpers.conditioning_set_values(c, {"reference_pixels": [pixE], "strength": float(pixE_w), "timestep_percent_range": [float(rng[0]), float(rng[1])]}, append=True)
            if pixM is not None and pixM_w > 0:
                c = node_helpers.conditioning_set_values(c, {"reference_pixels": [pixM], "strength": float(pixM_w), "timestep_percent_range": [float(rng[0]), float(rng[1])]}, append=True)
            if pixL is not None and pixL_w > 0:
                r = hfr if hfr is not None else rng
                c = node_helpers.conditioning_set_values(c, {"reference_pixels": [pixL], "strength": float(pixL_w), "timestep_percent_range": [float(r[0]), float(r[1])]}, append=True)
            return c

        cond = _add_ref(cond, early, lat_e, pixE_e, pixM_e, 0.0, None)
        cond = _add_ref(cond, late, lat_l, 0.0, pixM_l, pixL_l, hf_rng)

        latent = {"samples": lat,
                  "qi_pad": {"top": int(top), "bottom": int(bottom), "left": int(left), "right": int(right),
                             "orig_h": int(Htgt), "orig_w": int(Wtgt),
                             "compute_h": int(padded.shape[1]), "compute_w": int(padded.shape[2])}}
        return (cond, _ensure_rgb3(_bhwc(image)), latent)

NODE_CLASS_MAPPINGS = {"QI_RefEditEncode_Safe": QI_RefEditEncode_Safe}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_RefEditEncode_Safe": "Qwen一致性编辑编码器 — by wallen0322"}
