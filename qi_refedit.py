from __future__ import annotations
import torch
import torch.nn.functional as F
import node_helpers
import comfy.utils

_ALIGN_M = 32
_SAFE_MAX_PIX = 3_000_000
_VL_MAX_PIX = 1_400_000

def _smart_align(v: int, prefer: int = 16, fallback: int = 8) -> int:
    if v < fallback:
        return fallback
    
    v_prefer = ((v + prefer - 1) // prefer) * prefer
    v_fallback = ((v + fallback - 1) // fallback) * fallback
    
    if abs(v_prefer - v) <= prefer // 2:
        return v_prefer
    return v_fallback

def _compute_dimensions(input_w: int, input_h: int, out_w: int, out_h: int, max_pix: int = _SAFE_MAX_PIX):
    target_w = out_w if out_w > 0 else input_w
    target_h = out_h if out_h > 0 else input_h
    
    area = target_w * target_h
    if area > max_pix:
        scale = (max_pix / float(area)) ** 0.5
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)
    
    work_w = _smart_align(target_w, 16, 8)
    work_h = _smart_align(target_h, 16, 8)
    
    pad_w = _ceil_to(work_w, _ALIGN_M)
    pad_h = _ceil_to(work_h, _ALIGN_M)
    
    scale = min(work_w / float(input_w), work_h / float(input_h))
    letter_w = _smart_align(int(input_w * scale), 16, 8)
    letter_h = _smart_align(int(input_h * scale), 16, 8)
    letter_w = min(letter_w, work_w)
    letter_h = min(letter_h, work_h)
    
    return work_w, work_h, pad_w, pad_h, letter_w, letter_h

def _ceil_to(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m if m > 1 else v

def _floor_to(v: int, m: int) -> int:
    return (v // m) * m if m > 1 else v

def _round_to(v: int, m: int) -> int:
    return round(v / m) * m if m > 1 else v

def _to_bhwc_any(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, dict):
        for k in ("image","images","samples","data","result"):
            if k in x and isinstance(x[k], torch.Tensor):
                x = x[k]; break
    t = x
    if not isinstance(t, torch.Tensor):
        raise RuntimeError("Unsupported image container")
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(-1)
    elif t.ndim == 3:
        if t.shape[0] in (1,3,4):
            t = t.unsqueeze(0).movedim(1, -1)
        else:
            t = t.unsqueeze(0)
    elif t.ndim == 4:
        if t.shape[-1] in (1,3,4):
            pass
        elif t.shape[1] in (1,3,4):
            t = t.movedim(1, -1)
        else:
            t = t.movedim(1, -1)
    else:
        raise RuntimeError(f"Unsupported tensor shape for image: {tuple(t.shape)}")
    t = t.float()
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    if t.shape[-1] == 1:
        t = t.repeat(1,1,1,3)
    return t.clamp(0,1).contiguous()

def _ensure_rgb3(bhwc: torch.Tensor) -> torch.Tensor:
    if bhwc.shape[-1] == 3: return bhwc
    if bhwc.shape[-1] == 1: return bhwc.repeat(1,1,1,3)
    return bhwc[...,:3]

def _bhwc(x):
    return _ensure_rgb3(_to_bhwc_any(x))

def _bchw(x):
    return _bhwc(x).movedim(-1, 1).contiguous()

def _choose_method(sw, sh, tw, th):
    if tw*th <= 0: return "area"
    if tw >= sw or th >= sh: return "lanczos"
    return "area"

def _resize_bchw(x: torch.Tensor, w: int, h: int, method: str):
    return comfy.utils.common_upscale(x, w, h, method, "disabled")

def _resize_bchw_smart(x: torch.Tensor, w: int, h: int):
    _,_,H,W = x.shape
    return _resize_bchw(x, w, h, _choose_method(W,H,w,h))

def _letterbox_to_size(src_bhwc: torch.Tensor, letter_w: int, letter_h: int, work_w: int, work_h: int, pad_mode: str="reflect"):
    B, H, W, C = src_bhwc.shape
    
    if W == letter_w and H == letter_h and work_w == letter_w and work_h == letter_h:
        return src_bhwc, dict(top=0, bottom=0, left=0, right=0)
    
    base = _ensure_rgb3(_resize_bchw_smart(_bchw(src_bhwc), letter_w, letter_h).movedim(1, -1))
    
    top = (work_h - letter_h) // 2
    bottom = work_h - letter_h - top
    left = (work_w - letter_w) // 2
    right = work_w - letter_w - left
    
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        bchw = _bchw(base)
        h, w = bchw.shape[2], bchw.shape[3]
        mode = pad_mode
        if h < 2 or w < 2 or left >= w or right >= w or top >= h or bottom >= h:
            mode = "replicate"
        bchw = F.pad(bchw, (left, right, top, bottom), mode=mode)
        base = _ensure_rgb3(bchw.movedim(1, -1))
    
    return base, dict(top=int(top), bottom=int(bottom), left=int(left), right=int(right))

def _smoothstep(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    t = (x - a) / max(1e-6, (b - a))
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055)/1.055) ** 2.4)

def _linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.0031308, t*12.92, 1.055*torch.pow(t, 1.0/2.4) - 0.055)

def _chol2x2(a,b,c, eps=1e-8):
    a = torch.clamp(a, min=eps)
    l11 = torch.sqrt(a)
    l21 = b / l11
    t = torch.clamp(c - l21*l21, min=eps)
    l22 = torch.sqrt(t)
    inv_l11 = 1.0/l11
    inv_l22 = 1.0/l22
    inv_l21 = -l21 * inv_l11 * inv_l22
    return l11, l21, l22, inv_l11, inv_l21, inv_l22

def _apply_chroma_stats(Cbx, Crx, Cbr, Crr, Yx):
    B,_,H,W = Yx.shape
    yv = Yx.reshape(B,1,H*W)
    lo = torch.quantile(yv, 0.06, dim=-1, keepdim=True).reshape(B,1,1,1)
    hi = torch.quantile(yv, 0.94, dim=-1, keepdim=True).reshape(B,1,1,1)
    mask = (Yx>=lo) & (Yx<=hi)
    meanY = Yx.mean(dim=(2,3), keepdim=True)
    base_thr = 1.6e-3
    hi_thr = 2.0e-3
    thr = torch.full_like(Yx, base_thr)
    thr = torch.where(meanY > 0.72, torch.full_like(thr, hi_thr), thr)
    chroma2 = Cbx*Cbx + Crx*Crx
    mask = mask & (chroma2 > thr)
    m = mask.float()
    denom = m.sum(dim=(2,3), keepdim=True).clamp_min(1.0)

    def _stats(Cb, Cr):
        mu_cb = (Cb*m).sum(dim=(2,3), keepdim=True)/denom
        mu_cr = (Cr*m).sum(dim=(2,3), keepdim=True)/denom
        dcb = (Cb - mu_cb)*m
        dcr = (Cr - mu_cr)*m
        a = (dcb*(Cb - mu_cb)).sum(dim=(2,3), keepdim=True)/denom + 1e-6
        c = (dcr*(Cr - mu_cr)).sum(dim=(2,3), keepdim=True)/denom + 1e-6
        b = (dcb*(Cr - mu_cr)).sum(dim=(2,3), keepdim=True)/denom
        return mu_cb, mu_cr, a.squeeze(), b.squeeze(), c.squeeze()

    mu_x_cb, mu_x_cr, ax, bx, cx = _stats(Cbx, Crx)
    mu_r_cb, mu_r_cr, ar, br, cr = _stats(Cbr, Crr)
    l11x, l21x, l22x, inv11x, inv21x, inv22x = _chol2x2(ax, bx, cx)
    l11r, l21r, l22r, _, _, _ = _chol2x2(ar, br, cr)
    t00 = l11r*inv11x + 0.0*inv21x
    t01 = l11r*0.0    + 0.0*inv22x
    t10 = l21r*inv11x + l22r*inv21x
    t11 = l21r*0.0    + l22r*inv22x
    t00 = t00.clamp(0.985, 1.015)
    t11 = t11.clamp(0.985, 1.015)
    t01 = t01.clamp(-0.015, 0.015)
    t10 = t10.clamp(-0.015, 0.015)
    dmu_cb = (mu_r_cb - mu_x_cb)
    dmu_cr = (mu_r_cr - mu_x_cr)
    base_mu = 0.002
    strict_mu = 0.0015
    mu_lim = torch.where(meanY > 0.72, torch.full_like(meanY, strict_mu), torch.full_like(meanY, base_mu))
    dmu_cb = dmu_cb.clamp(-mu_lim, mu_lim)
    dmu_cr = dmu_cr.clamp(-mu_lim, mu_lim)
    delta = torch.clamp(meanY - 0.72, min=0.0, max=0.20)
    k = 1.0 - delta * 0.55
    k = torch.clamp(k, 0.75, 1.0)
    t00 = 1.0 + (t00 - 1.0)*k
    t11 = 1.0 + (t11 - 1.0)*k
    t01 = t01 * k
    t10 = t10 * k
    dmu_cb = dmu_cb * k
    dmu_cr = dmu_cr * k
    return (t00,t01,t10,t11), (dmu_cb,dmu_cr), (mu_x_cb,mu_x_cr)

def _color_lock_to(x_bhwc: torch.Tensor, ref_bhwc: torch.Tensor, mix: float=0.985) -> torch.Tensor:
    x = _bchw(x_bhwc); r = _bchw(ref_bhwc)
    x_lin = _srgb_to_linear(x); r_lin = _srgb_to_linear(r)
    R,G,B   = x_lin[:,0:1], x_lin[:,1:2], x_lin[:,2:3]
    Rr,Gr,Br= r_lin[:,0:1], r_lin[:,1:2], r_lin[:,2:3]
    Yx = 0.2126*R + 0.7152*G + 0.0722*B
    Yr = 0.2126*Rr + 0.7152*Gr + 0.0722*Br
    cb_s = 0.5/(1.0-0.0722); cr_s = 0.5/(1.0-0.2126)
    Cbx = (B - Yx)*cb_s; Crx = (R - Yx)*cr_s
    Cbr = (Br - Yr)*cb_s; Crr = (Rr - Yr)*cr_s
    (t00,t01,t10,t11), (dmu_cb,dmu_cr), (mu_x_cb,mu_x_cr) = _apply_chroma_stats(Cbx, Crx, Cbr, Crr, Yx)
    cb = Cbx - mu_x_cb; cr = Crx - mu_x_cr
    Cb_aligned = t00*cb + t01*cr + (mu_x_cb + dmu_cb)
    Cr_aligned = t10*cb + t11*cr + (mu_x_cr + dmu_cr)
    inv_cb = 1.0/cb_s; inv_cr = 1.0/cr_s
    R2 = Yx + Cr_aligned*inv_cr
    B2 = Yx + Cb_aligned*inv_cb
    G2 = (Yx - 0.2126*R2 - 0.0722*B2) / 0.7152
    aligned = torch.cat([R2,G2,B2], dim=1).clamp(0,1)
    w_hi = 1.0 - _smoothstep(0.74, 0.93, Yx)
    w_lo = _smoothstep(0.05, 0.12, Yx)
    w = (0.9 * mix * w_hi * w_lo).clamp(0.0, 1.0)
    out_lin = w*aligned + (1.0 - w)*x_lin
    out = _linear_to_srgb(out_lin).movedim(1,-1)
    return _ensure_rgb3(out)

def _simple_downsample(bchw: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    target_w = _smart_align(target_w, 16, 8)
    target_h = _smart_align(target_h, 16, 8)
    return _resize_bchw(bchw, target_w, target_h, "area")

def _lowpass_ref(bhwc: torch.Tensor, size: int=64) -> torch.Tensor:
    bchw = _bchw(bhwc)
    bh = F.interpolate(bchw, size=(size,size), mode="area")
    bh = _resize_bchw_smart(bh, bchw.shape[-1], bchw.shape[-2])
    return _ensure_rgb3(bh.movedim(1,-1)).clamp(0,1)

def _hf_ref_smart(bhwc: torch.Tensor, alpha: float, blur_k: int, kstd: float, bias: float, smooth_k: int) -> torch.Tensor:
    bchw = _bchw(bhwc)
    k = max(3, blur_k|1); pad = k//2
    if bchw.shape[2] <= pad or bchw.shape[3] <= pad:
        return _ensure_rgb3(bchw.movedim(1,-1))
    base = F.avg_pool2d(bchw, kernel_size=k, stride=1, padding=pad)
    detail = bchw - base
    ker = torch.ones((1,1,3,3), dtype=bchw.dtype, device=bchw.device)/9.0
    y = (_bchw(bhwc)[:,0:1]*0.299 + _bchw(bhwc)[:,1:2]*0.587 + _bchw(bhwc)[:,2:3]*0.114)
    mu = F.conv2d(F.pad(y,(1,1,1,1), mode="replicate"), ker)
    var= F.conv2d(F.pad((y-mu)**2,(1,1,1,1), mode="replicate"), ker)
    std= torch.sqrt(var + 1e-6)
    t  = kstd*std + bias
    absd = torch.abs(detail)
    t3 = torch.cat([t,t,t], dim=1)
    d = torch.sign(detail) * torch.clamp(absd - t3, min=0.0)
    sobel_x = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=y.dtype, device=y.device).view(1,1,3,3)/8.0
    sobel_y = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=y.dtype, device=y.device).view(1,1,3,3)/8.0
    e = torch.sqrt(F.conv2d(F.pad(y,(1,1,1,1),mode="replicate"), sobel_x)**2 +
                   F.conv2d(F.pad(y,(1,1,1,1),mode="replicate"), sobel_y)**2) + 1e-6
    mid = torch.exp(-((y-0.5)**2)/(2*0.25*0.25))
    tex = (std/(std.mean(dim=(2,3), keepdim=True)+1e-6)).clamp(0,1.5)
    tex = (tex - 0.2)/0.8; tex = tex.clamp(0,1)
    gate = (0.16 + 0.40*(e/(e.mean(dim=(2,3), keepdim=True)*3.0)) + 0.10*tex) * (0.85 + 0.15*mid)
    gate3 = torch.cat([gate,gate,gate], dim=1)
    dm = torch.tanh(d*2.0)*0.30
    out = (bchw + alpha*(dm*gate3)).clamp(0,1)
    return _ensure_rgb3(out.movedim(1,-1))

QUALITY_CONFIGS = {
    "fast": {
        "lat_e": 0.65, "pixM_e": 0.38, "pixE_e": 0.15,
        "lat_l": 0.92, "pixM_l": 0.75, "pixL_l": 0.016,
        "hf_alpha": 0.13, "kstd": 0.80, "bias": 0.0048
    },
    "best": {
        "lat_e": 0.72, "pixM_e": 0.48, "pixE_e": 0.19,
        "lat_l": 1.08, "pixM_l": 0.92, "pixL_l": 0.022,
        "hf_alpha": 0.16, "kstd": 0.88, "bias": 0.0045
    },
    "balanced": {
        "lat_e": 0.68, "pixM_e": 0.44, "pixE_e": 0.17,
        "lat_l": 1.00, "pixM_l": 0.85, "pixL_l": 0.019,
        "hf_alpha": 0.15, "kstd": 0.85, "bias": 0.0046
    },
    "natural": {
        "lat_e": 0.70, "pixM_e": 0.45, "pixE_e": 0.17,
        "lat_l": 1.05, "pixM_l": 0.88, "pixL_l": 0.020,
        "hf_alpha": 0.14, "kstd": 0.82, "bias": 0.005
    }
}

def _prepare_reference_images(images_in, vae):
    images_vl = []
    ref_latents = []
    image_prompt = ""
    
    for i, im in enumerate(images_in):
        if im is None:
            continue
        im_bhwc = _bhwc(im)[...,:3]
        samples = im_bhwc.movedim(-1, 1)
        Hs, Ws = samples.shape[2], samples.shape[3]
        
        total = int(384 * 384)
        scale_by = (total / float(Ws*Hs)) ** 0.5 if (Ws*Hs)>0 else 1.0
        width  = _smart_align(max(8, int(round(Ws * scale_by))), 16, 8)
        height = _smart_align(max(8, int(round(Hs * scale_by))), 16, 8)
        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        images_vl.append(s.movedim(1, -1))
        
        if vae is not None:
            total_l = int(1024 * 1024)
            scale_by_l = (total_l / float(Ws*Hs)) ** 0.5 if (Ws*Hs)>0 else 1.0
            width_l  = _smart_align(max(8, int(round(Ws * scale_by_l))), 16, 8)
            height_l = _smart_align(max(8, int(round(Hs * scale_by_l))), 16, 8)
            s_l = comfy.utils.common_upscale(samples, width_l, height_l, "area", "disabled")
            ref_latents.append(vae.encode(s_l.movedim(1,-1)[:, :, :, :3]))
        
        image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
    
    return images_vl, ref_latents, image_prompt

def _adjust_params_for_mode(cfg, emph, is_multi_image):
    params = cfg.copy()
    
    if is_multi_image:
        ref_scale = max(0.70, min(1.05, 1.05 - 0.35*emph))
        e_mult = max(0.50, min(1.10, 1.0 - 0.65 * emph))
        l_mult = max(0.75, min(1.10, 0.78 + 0.35 * (1.0 - emph)))
        
        params["pixM_e"] *= ref_scale * 0.92 * 0.85 * e_mult
        params["pixM_l"] *= ref_scale * 0.95 * l_mult
        params["pixE_e"] *= ref_scale * e_mult * 0.90
        params["pixL_l"] *= 0.95 * l_mult
    else:
        ref_scale = max(0.75, min(1.03, 1.03 - 0.28*emph))
        e_mult = max(0.75, min(1.05, 1.0 - 0.35 * emph))
        l_mult = max(0.85, min(1.08, 0.88 + 0.20 * (1.0 - emph)))
        
        params["pixM_e"] *= ref_scale * 0.85 * e_mult
        params["pixM_l"] *= ref_scale * l_mult
        params["pixE_e"] *= ref_scale * e_mult
        params["pixL_l"] *= 0.95 * l_mult
    
    return params

def _build_reference_pixels(padded, params, is_multi_image):
    pixM = pixE = pixL = None
    
    if is_multi_image:
        bchw_padded = _bchw(padded)
        Ph, Pw = bchw_padded.shape[2], bchw_padded.shape[3]
        total_pix_target = int(1024 * 1024)
        scale_for_ref = (total_pix_target / float(Pw * Ph)) ** 0.5
        Pw_ref = max(8, int(round(Pw * scale_for_ref)))
        Ph_ref = max(8, int(round(Ph * scale_for_ref)))
        
        padded_ref_bchw = _simple_downsample(bchw_padded, Pw_ref, Ph_ref)
        padded_ref = _ensure_rgb3(padded_ref_bchw.movedim(1, -1)).clamp(0, 1)
        
        if params["pixM_e"]>0 or params["pixM_l"]>0:
            pixM = _color_lock_to(padded_ref, padded_ref, mix=0.92)
        if params["pixE_e"]>0:
            pixE = _color_lock_to(_lowpass_ref(padded_ref, 64), padded_ref, mix=0.90)
    else:
        if params["pixM_e"]>0 or params["pixM_l"]>0:
            pixM = _color_lock_to(padded, padded, mix=0.985)
        if params["pixE_e"]>0:
            pixE = _color_lock_to(_lowpass_ref(padded, 64), padded, mix=0.99)
        if params["pixL_l"]>0:
            pixL = _color_lock_to(_hf_ref_smart(padded, params["hf_alpha"], 3, params["kstd"], params["bias"], 5), padded, mix=0.975)
    
    return pixM, pixE, pixL

def _add_references(cond, lat, refs, params, is_multi_image):
    pixM, pixE, pixL = refs
    
    if is_multi_image:
        ranges = {"early": (0.0, 0.6), "late": (0.6, 1.0)}
        pixE_e_val = params["pixE_e"] * 0.70
        if pixE is not None and pixE_e_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixE],
                "strength": float(min(pixE_e_val, 1.0)),
                "timestep_percent_range": [ranges["early"][0], ranges["early"][1]],
            }, append=True)
        lat_l_val = params["lat_l"] * 0.75
        pixM_l_val = params["pixM_l"] * 1.05
        if lat_l_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_latents": [lat],
                "strength": float(min(lat_l_val, 1.0)),
                "timestep_percent_range": [ranges["late"][0], ranges["late"][1]],
            }, append=True)
        if pixM is not None and pixM_l_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixM],
                "strength": float(min(pixM_l_val, 1.0)),
                "timestep_percent_range": [ranges["late"][0], ranges["late"][1]],
            }, append=True)
    else:
        ranges = {"early": (0.0, 0.35), "mid": (0.35, 0.75), "late": (0.75, 1.0)}
        lat_e_val = params["lat_e"] * 0.65
        pixE_e_val = params["pixE_e"] * 0.90
        pixM_e_val = params["pixM_e"] * 0.75
        
        if lat_e_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_latents": [lat],
                "strength": float(min(lat_e_val, 1.0)),
                "timestep_percent_range": [ranges["early"][0], ranges["early"][1]],
            }, append=True)
        if pixE is not None and pixE_e_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixE],
                "strength": float(min(pixE_e_val, 1.0)),
                "timestep_percent_range": [ranges["early"][0], ranges["early"][1]],
            }, append=True)
        if pixM is not None and pixM_e_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixM],
                "strength": float(min(pixM_e_val, 1.0)),
                "timestep_percent_range": [ranges["early"][0], ranges["early"][1]],
            }, append=True)
        
        lat_m_val = params["lat_l"] * 0.85
        pixM_m_val = params["pixM_l"] * 1.00
        pixL_m_val = params["pixL_l"] * 0.85
        
        if lat_m_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_latents": [lat],
                "strength": float(min(lat_m_val, 1.0)),
                "timestep_percent_range": [ranges["mid"][0], ranges["mid"][1]],
            }, append=True)
        if pixM is not None and pixM_m_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixM],
                "strength": float(min(pixM_m_val, 1.0)),
                "timestep_percent_range": [ranges["mid"][0], ranges["mid"][1]],
            }, append=True)
        if pixL is not None and pixL_m_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixL],
                "strength": float(min(pixL_m_val, 1.0)),
                "timestep_percent_range": [ranges["mid"][0], ranges["mid"][1]],
            }, append=True)
        
        lat_l_val = params["lat_l"] * 0.85
        pixM_l_val = params["pixM_l"] * 1.15
        pixL_l_val = params["pixL_l"] * 1.30
        
        if lat_l_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_latents": [lat],
                "strength": float(min(lat_l_val, 1.0)),
                "timestep_percent_range": [ranges["late"][0], ranges["late"][1]],
            }, append=True)
        if pixM is not None and pixM_l_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixM],
                "strength": float(min(pixM_l_val, 1.0)),
                "timestep_percent_range": [ranges["late"][0], ranges["late"][1]],
            }, append=True)
        if pixL is not None and pixL_l_val > 0.01:
            cond = node_helpers.conditioning_set_values(cond, {
                "reference_pixels": [pixL],
                "strength": float(min(pixL_l_val, 1.0)),
                "timestep_percent_range": [ranges["late"][0], ranges["late"][1]],
            }, append=True)
    
    return cond

class QI_RefEditEncode_Safe:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("CONDITIONING","IMAGE","LATENT")
    RETURN_NAMES = ("conditioning","image","latent")
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
            "prompt_emphasis": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "quality_mode": (["natural","fast","balanced","best"], {"default": "natural"}),
            "brightness_boost": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 1.3, "step": 0.01}),
            "debug_info": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "image2": ("IMAGE",),
            "image3": ("IMAGE",)
        }}

    def encode(self, clip, prompt, image, vae, out_width=0, out_height=0, prompt_emphasis=0.5, quality_mode="natural", brightness_boost=1.0, debug_info=False, image2=None, image3=None):
        src = _bhwc(image)[...,:3]
        H, W = int(src.shape[1]), int(src.shape[2])
        
        work_w, work_h, pad_w, pad_h, letter_w, letter_h = _compute_dimensions(W, H, int(out_width), int(out_height), _SAFE_MAX_PIX)
        
        letter, ext = _letterbox_to_size(src, letter_w, letter_h, work_w, work_h, pad_mode="reflect")
        top, left, bottom, right = ext["top"], ext["left"], ext["bottom"], ext["right"]
        
        ph, pw = pad_h - work_h, pad_w - work_w
        if ph > 0 or pw > 0:
            t2 = ph // 2
            b2 = ph - t2
            l2 = pw // 2
            r2 = pw - l2
            bchw = _bchw(letter)
            h, w = bchw.shape[2], bchw.shape[3]
            _mode = "reflect"
            if h < 2 or w < 2 or l2 >= w or r2 >= w or t2 >= h or b2 >= h:
                _mode = "replicate"
            bchw = F.pad(bchw, (l2, r2, t2, b2), mode=_mode)
            letter = _ensure_rgb3(bchw.movedim(1, -1))
            top += t2
            bottom += b2
            left += l2
            right += r2
        
        padded = letter.contiguous()
        
        images_in = [image, image2, image3]
        num_images = sum(1 for im in images_in if im is not None)
        is_multi_image = num_images > 1
        
        images_vl, ref_latents, image_prompt = _prepare_reference_images(images_in, vae)
        
        chat = "<|im_start|>user\n" + image_prompt + (prompt if isinstance(prompt,str) else str(prompt)) + "\n<|im_end|>\n<|im_start|>assistant\n"
        tokens = clip.tokenize(chat, images=images_vl)
        cond = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            cond = node_helpers.conditioning_set_values(cond, {"reference_latents": ref_latents}, append=True)
        
        padded_for_vae = padded
        if brightness_boost != 1.0:
            gamma = 1.0 / brightness_boost
            padded_for_vae = torch.pow(padded.clamp(0, 1), gamma).clamp(0, 1)
        
        with torch.inference_mode():
            lat = vae.encode(padded_for_vae)
            if isinstance(lat, dict) and "samples" in lat: 
                lat = lat["samples"]
            if isinstance(lat, torch.Tensor) and lat.dtype != torch.float32: 
                lat = lat.float()
        
        cfg = QUALITY_CONFIGS[quality_mode]
        emph = float(max(0.0, min(1.0, prompt_emphasis)))
        params = _adjust_params_for_mode(cfg, emph, is_multi_image)
        
        pixM, pixE, pixL = _build_reference_pixels(padded, params, is_multi_image)
        
        cond = _add_references(cond, lat, (pixM, pixE, pixL), params, is_multi_image)
        
        if debug_info:
            print(f"\n=== Qwen Edit Debug ===")
            print(f"Input: {W}x{H} → Work: {work_w}x{work_h} → Pad: {pad_w}x{pad_h}")
            print(f"Letter: {letter_w}x{letter_h}")
            print(f"Mode: {'Multi' if is_multi_image else 'Single'} ({num_images} imgs)")
            print(f"Quality: {quality_mode}, Emphasis: {emph:.2f}, Brightness: {brightness_boost:.2f}")
            print(f"Params: lat_e={params['lat_e']:.2f} lat_l={params['lat_l']:.2f}")
            print(f"        pixM_e={params['pixM_e']:.2f} pixM_l={params['pixM_l']:.2f}")
            print(f"        pixE_e={params['pixE_e']:.2f} pixL_l={params['pixL_l']:.2f}")
            print(f"Refs: M={'Y' if pixM is not None else 'N'} E={'Y' if pixE is not None else 'N'} L={'Y' if pixL is not None else 'N'}")
            print(f"======================\n")
        
        latent = {
            "samples": lat,
            "qi_pad": {
                "top": int(top), "bottom": int(bottom), 
                "left": int(left), "right": int(right),
                "orig_h": int(work_h), "orig_w": int(work_w),
                "compute_h": int(padded.shape[1]), "compute_w": int(padded.shape[2])
            }
        }
        
        return (cond, _ensure_rgb3(_bhwc(image)), latent)

NODE_CLASS_MAPPINGS = {"QI_RefEditEncode_Safe": QI_RefEditEncode_Safe}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_RefEditEncode_Safe": "Qwen一致性编辑编码器 – by wallen0322"}
