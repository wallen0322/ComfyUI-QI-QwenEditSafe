from __future__ import annotations
import torch
import torch.nn.functional as F
import node_helpers, comfy.utils

# -------------------- constants --------------------
_ALIGN_M = 32
_SAFE_MAX_PIX = 3_000_000
_VL_MAX_PIX = 1_400_000

# -------------------- basic image helpers --------------------
def _ceil_to(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m if m > 1 else v

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

def _letterbox(src_bhwc: torch.Tensor, Wt: int, Ht: int, pad_mode: str="reflect"):
    B,H,W,C = src_bhwc.shape
    if W==Wt and H==Ht:
        return src_bhwc, dict(top=0,bottom=0,left=0,right=0)
    s = min(Wt/float(W), Ht/float(H))
    Wr = max(1, int(round(W*s))); Hr = max(1, int(round(H*s)))
    base = _ensure_rgb3(_resize_bchw_smart(_bchw(src_bhwc), Wr, Hr).movedim(1,-1))
    top = (Ht - Hr)//2; bottom = Ht-Hr-top
    left = (Wt - Wr)//2; right  = Wt-Wr-left
    bchw = _bchw(base)
    if top>0 or bottom>0 or left>0 or right>0:
        h, w = int(bchw.shape[2]), int(bchw.shape[3])
        mode = pad_mode
        if h < 2 or w < 2 or left >= w or right >= w or top >= h or bottom >= h:
            mode = "replicate"
        bchw = F.pad(bchw, (left,right,top,bottom), mode=mode)
    return _ensure_rgb3(bchw.movedim(1,-1)), dict(top=int(top),bottom=int(bottom),left=int(left),right=int(right))

def _cap_vl(img_bhwc: torch.Tensor) -> torch.Tensor:
    H,W = int(img_bhwc.shape[1]), int(img_bhwc.shape[2])
    area = H*W
    if area <= _VL_MAX_PIX: return img_bhwc
    s = (_VL_MAX_PIX/float(area))**0.5
    Hs = max(1,int(H*s)); Ws = max(1,int(W*s))
    return _ensure_rgb3(_resize_bchw_smart(_bchw(img_bhwc), Ws, Hs).movedim(1,-1))

# -------------------- color helpers (Linear BT.709) --------------------
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

def _color_lock_to(x_bhwc: torch.Tensor, ref_bhwc: torch.Tensor, mix: float=0.97) -> torch.Tensor:
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

# -------------------- reference builders --------------------
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

# -------------------- node class --------------------
def _build_chatml_multiimage(user_text: str, n_images: int) -> str:
    parts = ["<|im_start|>user"]
    for _ in range(max(0, int(n_images))):
        parts.append("<|vision_start|><|image_pad|><|vision_end|>")
    if user_text and len(user_text.strip())>0:
        parts.append(user_text.strip())
    parts.append("<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)

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
            "prompt_emphasis": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            "quality_mode": (["natural","fast","balanced","best"], {"default": "natural"}),
        },
        "optional": {
            "image2": ("IMAGE",),
            "image3": ("IMAGE",)
        }}

    def encode(self, clip, prompt, image, vae, out_width=0, out_height=0, prompt_emphasis=0.6, quality_mode="natural", image2=None, image3=None):
        src = _bhwc(image)[...,:3]
        H,W = int(src.shape[1]), int(src.shape[2])
        Wt = int(out_width) if int(out_width)>0 else W
        Ht = int(out_height) if int(out_height)>0 else H

        area = Wt*Ht
        if area > _SAFE_MAX_PIX:
            s = (_SAFE_MAX_PIX/float(area))**0.5
            Wt = max(8, int(Wt*s)); Ht = max(8, int(Ht*s))

        Wc = _ceil_to(Wt, _ALIGN_M); Hc = _ceil_to(Ht, _ALIGN_M)

        letter, ext = _letterbox(src, Wt, Ht, pad_mode="reflect")
        top,left,bottom,right = ext["top"],ext["left"],ext["bottom"],ext["right"]
        ph, pw = Hc - Ht, Wc - Wt
        if ph>0 or pw>0:
            t2 = ph//2; b2 = ph - t2; l2 = pw//2; r2 = pw - l2
            bchw = _bchw(letter)
            h, w = int(bchw.shape[2]), int(bchw.shape[3])
            _mode = "reflect"
            if h < 2 or w < 2 or l2 >= w or r2 >= w or t2 >= h or b2 >= h:
                _mode = "replicate"
            bchw = F.pad(bchw, (l2,r2,t2,b2), mode=_mode)
            letter = _ensure_rgb3(bchw.movedim(1,-1))
            top+=t2; bottom+=b2; left+=l2; right+=r2
        padded = letter.contiguous()

        images_vl = []
        ref_latents = []
        images_in = [image, image2, image3]
        image_prompt = ""
        for i, im in enumerate(images_in):
            if im is None:
                continue
            im_bhwc = _bhwc(im)[...,:3]
            samples = im_bhwc.movedim(-1, 1)
            total = int(384 * 384)
            Hs, Ws = samples.shape[2], samples.shape[3]
            scale_by = (total / float(Ws*Hs)) ** 0.5 if (Ws*Hs)>0 else 1.0
            width  = max(1, int(round(Ws * scale_by)))
            height = max(1, int(round(Hs * scale_by)))
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1))
            if vae is not None:
                total_l = int(1024 * 1024)
                scale_by_l = (total_l / float(Ws*Hs)) ** 0.5 if (Ws*Hs)>0 else 1.0
                width_l  = max(8, int(round(Ws * scale_by_l / 8.0)) * 8)
                height_l = max(8, int(round(Hs * scale_by_l / 8.0)) * 8)
                s_l = comfy.utils.common_upscale(samples, width_l, height_l, "area", "disabled")
                ref_latents.append(vae.encode(s_l.movedim(1,-1)[:, :, :, :3]))
            image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
        chat = "<|im_start|>user\n" + image_prompt + (prompt if isinstance(prompt,str) else str(prompt)) + "\n<|im_end|>\n<|im_start|>assistant\n"
        tokens = clip.tokenize(chat, images=images_vl)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            cond = node_helpers.conditioning_set_values(cond, {"reference_latents": ref_latents}, append=True)

        with torch.inference_mode():
            lat = vae.encode(padded)
            if isinstance(lat, dict) and "samples" in lat: lat = lat["samples"]
            if isinstance(lat, torch.Tensor) and lat.dtype != torch.float32: lat = lat.float()

        emph = float(max(0.0, min(1.0, prompt_emphasis)))
        if quality_mode == "fast":
            lat_e, pixM_e, pixE_e = 0.65, 0.38, 0.15
            lat_l, pixM_l, pixL_l = 0.92, 0.75, 0.016
            hf_alpha, kstd, bias = 0.13, 0.80, 0.0048
        elif quality_mode == "best":
            lat_e, pixM_e, pixE_e = 0.72, 0.48, 0.19
            lat_l, pixM_l, pixL_l = 1.08, 0.92, 0.022
            hf_alpha, kstd, bias = 0.16, 0.88, 0.0045
        elif quality_mode == "balanced":
            lat_e, pixM_e, pixE_e = 0.68, 0.44, 0.17
            lat_l, pixM_l, pixL_l = 1.00, 0.85, 0.019
            hf_alpha, kstd, bias = 0.15, 0.85, 0.0046
        else:
            lat_e, pixM_e, pixE_e = 0.70, 0.45, 0.17
            lat_l, pixM_l, pixL_l = 1.05, 0.88, 0.020
            hf_alpha, kstd, bias = 0.14, 0.82, 0.005

        ref_scale = max(0.75, min(1.03, 1.03 - 0.28*emph))
        pixM_e *= ref_scale; pixM_l *= ref_scale; pixE_e *= ref_scale

        e_mult = max(0.75, min(1.05, 1.0 - 0.35 * emph))
        l_mult = max(0.85, min(1.08, 0.88 + 0.20 * (1.0 - emph)))

        pixE_e *= e_mult
        pixM_e *= 0.85 * e_mult
        pixM_l *= l_mult
        pixL_l *= 0.95 * l_mult

        pixM = padded if (pixM_e>0 or pixM_l>0) else None
        pixE = _lowpass_ref(padded, 64) if pixE_e>0 else None
        pixL = _hf_ref_smart(padded, hf_alpha, 3, kstd, bias, 5) if pixL_l>0 else None

        if pixM is not None: pixM = _color_lock_to(pixM, padded, mix=0.985)
        if pixE is not None: pixE = _color_lock_to(pixE, padded, mix=0.99)
        if pixL is not None: pixL = _color_lock_to(pixL, padded, mix=0.975)

        early = (0.0, 0.35); mid = (0.35, 0.75); late = (0.75, 1.0); hf_rng = (0.985, 1.0)

        def _add_ref(c, rng, lat_w, pixE_w, pixM_w, pixL_w, hfr=None):
            c = node_helpers.conditioning_set_values(c, {
                "reference_latents": [lat],
                "strength": float(lat_w),
                "timestep_percent_range": [float(rng[0]), float(rng[1])],
            }, append=True)
            if pixE is not None and pixE_w>0:
                c = node_helpers.conditioning_set_values(c, {
                    "reference_pixels": [pixE],
                    "strength": float(pixE_w),
                    "timestep_percent_range": [float(rng[0]), float(rng[1])],
                }, append=True)
            if pixM is not None and pixM_w>0:
                c = node_helpers.conditioning_set_values(c, {
                    "reference_pixels": [pixM],
                    "strength": float(pixM_w),
                    "timestep_percent_range": [float(rng[0]), float(rng[1])],
                }, append=True)
            if pixL is not None and pixL_w>0:
                r = hfr if hfr is not None else rng
                c = node_helpers.conditioning_set_values(c, {
                    "reference_pixels": [pixL],
                    "strength": float(pixL_w),
                    "timestep_percent_range": [float(r[0]), float(r[1])],
                }, append=True)
            return c

        cond = _add_ref(cond, early, lat_e*0.70, pixE_e*0.85, pixM_e*0.65, 0.0, None)
        cond = _add_ref(cond, mid, lat_l*0.95, 0.0, pixM_l*0.90, pixL_l*0.80, None)
        cond = _add_ref(cond, late, lat_l*1.15, 0.0, pixM_l*1.00, pixL_l*1.20, hf_rng)

        latent = {"samples": lat,
                  "qi_pad": {"top": int(top), "bottom": int(bottom), "left": int(left), "right": int(right),
                             "orig_h": int(Ht), "orig_w": int(Wt),
                             "compute_h": int(padded.shape[1]), "compute_w": int(padded.shape[2])}}
        return (cond, _ensure_rgb3(_bhwc(image)), latent)

NODE_CLASS_MAPPINGS = {"QI_RefEditEncode_Safe": QI_RefEditEncode_Safe}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_RefEditEncode_Safe": "Qwen一致性编辑编码器 – by wallen0322"}
