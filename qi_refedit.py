from __future__ import annotations
import torch
import torch.nn.functional as F
import re
import node_helpers, comfy.utils



# -------------------- constants --------------------
_ALIGN_M = 32                 # internal compute multiple
_SAFE_MAX_PIX = 3_000_000     # hard cap for compute (3MP)
_VL_MAX_PIX = 1_400_000       # visual input cap for CLIP vision (1.4MP)

# -------------------- basic image helpers --------------------
def _ceil_to(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m if m > 1 else v

def _to_bhwc_any(x):
    # accept Tensor or (list/tuple/dict)->first tensor
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, dict):
        for k in ("image","images","samples","data","result"):
            if k in x and isinstance(x[k], torch.Tensor):
                x = x[k]; break
    t = x
    if not isinstance(t, torch.Tensor):
        raise RuntimeError("Unsupported image container")
    if t.ndim == 2:      # H,W -> B,H,W,C=1
        t = t.unsqueeze(0).unsqueeze(-1)
    elif t.ndim == 3:    # CHW or HWC -> BHWC
        if t.shape[0] in (1,3,4):
            t = t.unsqueeze(0).movedim(1, -1)
        else:
            t = t.unsqueeze(0)
    elif t.ndim == 4:    # BCHW or BHWC -> BHWC
        if t.shape[-1] in (1,3,4):
            pass
        elif t.shape[1] in (1,3,4):
            t = t.movedim(1, -1)
        else:
            t = t.movedim(1, -1)
    else:
        raise RuntimeError(f"Unsupported tensor shape for image: {tuple(t.shape)}")
    t = t.float()
    if t.min().item() < 0.0:   # [-1,1] -> [0,1]
        t = (t + 1.0) * 0.5
    if t.shape[-1] == 1:
        t = t.repeat(1,1,1,3)
    return t.clamp(0,1).contiguous()

def _ensure_rgb3(bhwc: torch.Tensor) -> torch.Tensor:
    if bhwc.shape[-1] == 3: return bhwc
    if bhwc.shape[-1] == 1: return bhwc.repeat(1,1,1,3)
    return bhwc[...,:3]

def _bhwc(x):  # IMAGE -> BHWC
    return _ensure_rgb3(_to_bhwc_any(x))

def _bchw(x):  # IMAGE -> BCHW
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

# 2x2 Cholesky for SPD [[a,b],[b,c]]
def _chol2x2(a,b,c, eps=1e-8):
    a = torch.clamp(a, min=eps)
    l11 = torch.sqrt(a)
    l21 = b / l11
    t = torch.clamp(c - l21*l21, min=eps)
    l22 = torch.sqrt(t)
    # L = [[l11,0],[l21,l22]], invL:
    inv_l11 = 1.0/l11
    inv_l22 = 1.0/l22
    inv_l21 = -l21 * inv_l11 * inv_l22
    return l11, l21, l22, inv_l11, inv_l21, inv_l22

def _apply_chroma_stats(Cbx, Crx, Cbr, Crr, Yx):
    # robust midtone mask & exclude near-gray
    B,_,H,W = Yx.shape
    yv = Yx.reshape(B,1,H*W)
    lo = torch.quantile(yv, 0.06, dim=-1, keepdim=True).reshape(B,1,1,1)
    hi = torch.quantile(yv, 0.94, dim=-1, keepdim=True).reshape(B,1,1,1)
    mask = (Yx>=lo) & (Yx<=hi)
    # exclude very low chroma (brighter scenes stricter)
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

    # Cholesky whiten->color: T = Lr @ inv(Lx)
    l11x, l21x, l22x, inv11x, inv21x, inv22x = _chol2x2(ax, bx, cx)
    l11r, l21r, l22r, _, _, _ = _chol2x2(ar, br, cr)

    # inv(Lx) = [[inv11x, 0],[inv21x, inv22x]]
    # Lr = [[l11r,0],[l21r,l22r]]
    t00 = l11r*inv11x + 0.0*inv21x
    t01 = l11r*0.0    + 0.0*inv22x
    t10 = l21r*inv11x + l22r*inv21x
    t11 = l21r*0.0    + l22r*inv22x

    # clamp transform: tighten rotation & mean shifts for color fidelity
    t00 = t00.clamp(0.985, 1.015)   # diag shrink range
    t11 = t11.clamp(0.985, 1.015)
    t01 = t01.clamp(-0.015, 0.015)  # off-diag smaller
    t10 = t10.clamp(-0.015, 0.015)

    # mean shift clamp (brighter scenes stricter)
    dmu_cb = (mu_r_cb - mu_x_cb)
    dmu_cr = (mu_r_cr - mu_x_cr)
    base_mu = 0.002
    strict_mu = 0.0015
    mu_lim = torch.where(meanY > 0.72, torch.full_like(meanY, strict_mu), torch.full_like(meanY, base_mu))
    dmu_cb = dmu_cb.clamp(-mu_lim, mu_lim)
    dmu_cr = dmu_cr.clamp(-mu_lim, mu_lim)

    # brightness-adaptive conservative factor for very bright scenes
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
    # sRGB -> linear
    x = _bchw(x_bhwc); r = _bchw(ref_bhwc)
    x_lin = _srgb_to_linear(x); r_lin = _srgb_to_linear(r)

    R,G,B   = x_lin[:,0:1], x_lin[:,1:2], x_lin[:,2:3]
    Rr,Gr,Br= r_lin[:,0:1], r_lin[:,1:2], r_lin[:,2:3]

    # BT.709 YCbCr (linear)
    Yx = 0.2126*R + 0.7152*G + 0.0722*B
    Yr = 0.2126*Rr + 0.7152*Gr + 0.0722*Br
    cb_s = 0.5/(1.0-0.0722); cr_s = 0.5/(1.0-0.2126)
    Cbx = (B - Yx)*cb_s; Crx = (R - Yx)*cr_s
    Cbr = (Br - Yr)*cb_s; Crr = (Rr - Yr)*cr_s

    # 2x2 stats & transform
    (t00,t01,t10,t11), (dmu_cb,dmu_cr), (mu_x_cb,mu_x_cr) = _apply_chroma_stats(Cbx, Crx, Cbr, Crr, Yx)

    # apply per-pixel: v' = T (v - mu_x) + (mu_x + dmu)
    cb = Cbx - mu_x_cb; cr = Crx - mu_x_cr
    Cb_aligned = t00*cb + t01*cr + (mu_x_cb + dmu_cb)
    Cr_aligned = t10*cb + t11*cr + (mu_x_cr + dmu_cr)

    # reconstruct RGB (preserve Y)
    inv_cb = 1.0/cb_s; inv_cr = 1.0/cr_s
    R2 = Yx + Cr_aligned*inv_cr
    B2 = Yx + Cb_aligned*inv_cb
    G2 = (Yx - 0.2126*R2 - 0.0722*B2) / 0.7152
    aligned = torch.cat([R2,G2,B2], dim=1).clamp(0,1)

    # highlight/lowlight attenuation for blending — slightly more conservative mid
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
    # texture/edge gates
    sobel_x = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=y.dtype, device=y.device).view(1,1,3,3)/8.0
    sobel_y = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=y.dtype, device=y.device).view(1,1,3,3)/8.0
    e = torch.sqrt(F.conv2d(F.pad(y,(1,1,1,1),mode="replicate"), sobel_x)**2 +
                   F.conv2d(F.pad(y,(1,1,1,1),mode="replicate"), sobel_y)**2) + 1e-6
    mid = torch.exp(-((y-0.5)**2)/(2*0.25*0.25))
    tex = (std/(std.mean(dim=(2,3), keepdim=True)+1e-6)).clamp(0,1.5)
    tex = (tex - 0.2)/0.8; tex = tex.clamp(0,1)
    gate = (0.16 + 0.40*(e/(e.mean(dim=(2,3), keepdim=True)*3.0)) + 0.10*tex) * (0.85 + 0.15*mid)
    gate3 = torch.cat([gate,gate,gate], dim=1)
    dm = torch.tanh(d*2.0)*0.30  # slightly lower amplitude to avoid over-contrast
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
        # source/output sizes
        src = _bhwc(image)[...,:3]
        H,W = int(src.shape[1]), int(src.shape[2])
        Wt = int(out_width) if int(out_width)>0 else W
        Ht = int(out_height) if int(out_height)>0 else H

        # hard compute cap
        area = Wt*Ht
        if area > _SAFE_MAX_PIX:
            s = (_SAFE_MAX_PIX/float(area))**0.5
            Wt = max(8, int(Wt*s)); Ht = max(8, int(Ht*s))

        Wc = _ceil_to(Wt, _ALIGN_M); Hc = _ceil_to(Ht, _ALIGN_M)

        # letterbox -> pad to compute size
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

        # CLIP tokens with VL cap
        
        # ---- Official-like front half (Qwen TextEncodeQwenImageEditPlus) ----
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
        # per-image ref_latents will be appended with routed strengths below
        

        # ---- Routed multi-image helpers (minimal, no extra deps) ----
        _CLAUSE_SPLIT_RE = re.compile(r"[，,。.;；、]+|\band\b|\b并且\b|\b然后\b", re.IGNORECASE)

        def _split_prompt_into_clauses(text):
            try:
                parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(str(text)) if p.strip()]
                return parts if parts else [str(text)]
            except Exception:
                return [str(text)]

        _AGE_OLD = ("老人","老爷爷","长者","elder","elderly","old man","grandpa")
        _AGE_YOUNG = ("女孩","小女孩","姑娘","young","girl","young woman","little girl")

        def _prior_from_keywords(clause, n_imgs):
            g = [0.0]*n_imgs
            t = clause.lower()
            for i in range(n_imgs):
                g[i] = 0.0
            # weak priors: bias img1 if 'old/老人' appears; bias img2 if 'girl/女孩'
            # Note: keep tiny to avoid oversteer; acts only when similarity unavailable (we don't compute embeddings here).
            for k in _AGE_OLD:
                if k in clause:
                    if n_imgs>=1: g[0] += 0.15
            for k in _AGE_YOUNG:
                if k in clause.lower() or k in clause:
                    if n_imgs>=2: g[1] += 0.15
            return g

        def _uniform_alpha(n):
            v = [1.0/float(max(1,n))]*n
            return v

        def _alpha_from_order(idx, n):
            # simple round-robin mapping clause -> image by order as a fallback
            v = [0.0]*n
            if n>0:
                v[idx % n] = 1.0
            return v

        def _norm(v):
            s = float(sum(max(0.0,x) for x in v))
            if s<=1e-8: 
                n = len(v)
                return [1.0/float(max(1,n))]*n
            return [max(0.0,x)/s for x in v]
        # ---- End helpers ----

# ---- End official-like front half ----


        # VAE encode
        with torch.inference_mode():
            lat = vae.encode(padded)
            if isinstance(lat, dict) and "samples" in lat: lat = lat["samples"]
            if isinstance(lat, torch.Tensor) and lat.dtype != torch.float32: lat = lat.float()

        # schedule weights (quality presets)
        emph = float(max(0.0, min(1.0, prompt_emphasis)))
        if quality_mode == "fast":
            lat_e, pixM_e, pixE_e = 0.68, 0.40, 0.16; lat_l, pixM_l, pixL_l = 0.95, 0.78, 0.018; hf_alpha, kstd, bias, smooth = 0.14, 0.84, 0.0045, 5
        elif quality_mode == "best":
            lat_e, pixM_e, pixE_e = 0.76, 0.50, 0.20; lat_l, pixM_l, pixL_l = 1.05, 0.90, 0.020; hf_alpha, kstd, bias, smooth = 0.17, 0.90, 0.0045, 7
        elif quality_mode == "balanced":
            lat_e, pixM_e, pixE_e = 0.72, 0.46, 0.18; lat_l, pixM_l, pixL_l = 1.00, 0.85, 0.020; hf_alpha, kstd, bias, smooth = 0.16, 0.88, 0.0045, 5
        else:
            lat_e, pixM_e, pixE_e = 0.73, 0.47, 0.18; lat_l, pixM_l, pixL_l = 1.02, 0.86, 0.020; hf_alpha, kstd, bias, smooth = 0.16, 0.88, 0.0045, 5

        # global scale by prompt_emphasis (keep latent always-locked)
        ref_scale = max(0.70, min(1.05, 1.05 - 0.35*emph))
        pixM_e *= ref_scale; pixM_l *= ref_scale; pixE_e *= ref_scale

        # ---- two-step adherence: early "more prompt", late "more consistency" ----
        e_mult = max(0.5, min(1.1, 1.0 - 0.65 * emph))              # early: loosen pixels -> more prompt-following
        l_mult = max(0.75, min(1.10, 0.78 + 0.35 * (1.0 - emph)))   # late : tighten pixels -> stronger consistency (slightly milder)

        pixE_e *= e_mult
        pixM_e *= 0.85 * e_mult
        pixM_l *= l_mult
        pixL_l *= 0.95 * l_mult   # reduce micro-contrast in late HF

        # build references
        pixM = padded if (pixM_e>0 or pixM_l>0) else None
        pixE = _lowpass_ref(padded, 64) if pixE_e>0 else None
        pixL = _hf_ref_smart(padded, hf_alpha, 3, kstd, 0.0045, smooth) if pixL_l>0 else None

        # color-lock to original (reduce drift)
        if pixM is not None: pixM = _color_lock_to(pixM, padded, mix=0.97)
        if pixE is not None: pixE = _color_lock_to(pixE, padded, mix=0.97)
        if pixL is not None: pixL = _color_lock_to(pixL, padded, mix=0.97)

        early = (0.0, 0.6); late = (0.52, 1.0); hf_rng = (0.985, 1.0)

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

        # Step A (edit): routed per-image anchors for better prompt-role adherence


        lat_l_enh  = float(lat_l * 1.10)


        pixM_l_enh = float(pixM_l * 0.90)


        pixL_l_enh = float(pixL_l * 1.12)


        # Prepare per-image padded references aligned to compute size


        per_pixM, per_pixL = [], []


        for im in images_in:


            if im is None:


                per_pixM.append(None); per_pixL.append(None); continue


            im_bhwc = _bhwc(im)[...,:3]


            lb, _pad = _letterbox(im_bhwc, int(Wc), int(Hc))


            pixM_i = lb if (pixM_l_enh>0) else None


            pixE_i = None  # E disabled in ref-edit late strategy


            pixL_i = _hf_ref_smart(lb, hf_alpha, 3, kstd, 0.0045, smooth) if (pixL_l_enh>0) else None


            if pixM_i is not None: pixM_i = _color_lock_to(pixM_i, lb, mix=0.94)


            if pixL_i is not None: pixL_i = _color_lock_to(pixL_i, lb, mix=0.97)


            per_pixM.append(pixM_i); per_pixL.append(pixL_i)



        # Clause routing (lightweight heuristic: order + weak keywords)


        clauses = _split_prompt_into_clauses(prompt)


        n_img = len([im for im in images_in if im is not None])


        alpha = [0.0]*n_img


        for idx, c in enumerate(clauses):


            base = _alpha_from_order(idx, n_img)


            prior = _prior_from_keywords(c, n_img)


            mix = [b + p for b,p in zip(base, prior)]


            mix = _norm(mix)


            # accumulate per-image contribution


            alpha = [a + m for a, m in zip(alpha, mix)]


        # normalize accumulated weights


        alpha = _norm(alpha)






        # --- Consistency blend: interpolate routed weights with uniform to preserve identity across all refs ---


        n_active = max(1, n_img)


        min_floor = 0.22 if n_active >= 3 else (0.28 if n_active==2 else 1.0)


        beta = [max(float(a), float(min_floor)) for a in alpha]  # ensure each image has a floor share


        s_beta = float(sum(beta)) if sum(beta)>1e-8 else float(n_active)


        beta = [b/s_beta for b in beta]


        u = [1.0/float(n_active)]*n_active


        gamma = 0.58  # stronger consistency bias (0.0=text-route only, 1.0=uniform only)


        w = [(1.0-gamma)*beta[j] + gamma*u[j] for j in range(n_active)]


        # re-normalize to be safe


        sw = float(sum(w)) if sum(w)>1e-8 else float(n_active)


        w = [x/sw for x in w]


        # Strength normalization across images keeps total energy constant


        # Add per-image latents in a later window (shifted late)


        lat_rng = (0.50, 0.995)


        for j in range(n_img):


            if j < len(ref_latents):


                cond = node_helpers.conditioning_set_values(cond, {


                    "reference_latents": [ref_latents[j]],
            "strength": float(lat_l_enh * 1.24 * w[j]),


                    "timestep_percent_range": [float(lat_rng[0]), float(lat_rng[1])],


                }, append=True)



        # Add per-image mid-frequency pixels in a narrow window


        pixM_rng = (0.60, 0.78)


        for j in range(n_img):


            pm = per_pixM[j] if j < len(per_pixM) else None


            if pm is not None:


                cond = node_helpers.conditioning_set_values(cond, {


                    "reference_pixels": [pm],
            "strength": float(pixM_l_enh * 1.02 * w[j]),


                    "timestep_percent_range": [float(pixM_rng[0]), float(pixM_rng[1])],


                }, append=True)



        # Add per-image high-frequency pixels at very late window


        for j in range(n_img):


            pl = per_pixL[j] if j < len(per_pixL) else None


            if pl is not None:


                r = hf_rng


                cond = node_helpers.conditioning_set_values(cond, {


                    "reference_pixels": [pl],
            "strength": float(pixL_l_enh * 1.22 * w[j]),


                    "timestep_percent_range": [float(r[0]), float(r[1])],


                }, append=True)




        # === Ultra-late per-image consolidation (consistency-preserving, symmetric) ===




        if isinstance(w, list) and len(w)>0:




            n_active = max(1, n_img)




            # Build low-frequency color anchors per image (stronger lowpass for tone cohesion)




            per_pixC = []




            for im in images_in:




                if im is None:




                    per_pixC.append(None); continue




                im_bhwc = _bhwc(im)[...,:3]




                lb, _pad = _letterbox(im_bhwc, int(Wc), int(Hc))




                pixC_i = _lowpass_ref(lb, 128)




                pixC_i = _color_lock_to(pixC_i, lb, mix=0.985)




                per_pixC.append(pixC_i)




        




            ultra_late = (0.995, 1.0)




            # Inject per-image color anchors symmetrically




            for j in range(n_active):




                pc = per_pixC[j] if j < len(per_pixC) else None




                if pc is not None:




                    cond = node_helpers.conditioning_set_values(cond, {




                        "reference_pixels": [pc],




                        "strength": float(0.20 * w[j]),




                        "timestep_percent_range": [float(ultra_late[0]), float(ultra_late[1])],




                    }, append=True)




        




            # Ultra-late HF touch for all images (tiny but symmetric)




            for j in range(n_active):




                pl = per_pixL[j] if j < len(per_pixL) else None




                if pl is not None:




                    cond = node_helpers.conditioning_set_values(cond, {




                        "reference_pixels": [pl],




                        "strength": float(0.30 * w[j]),




                        "timestep_percent_range": [0.997, 1.0],




                    }, append=True)




        



            # Build low-frequency color anchors per image



            per_pixC = []



            for im in images_in:



                if im is None:



                    per_pixC.append(None); continue



                im_bhwc = _bhwc(im)[...,:3]



                lb, _pad = _letterbox(im_bhwc, int(Wc), int(Hc))



                pixC_i = _lowpass_ref(lb, 96)



                pixC_i = _color_lock_to(pixC_i, lb, mix=0.98)



                per_pixC.append(pixC_i)



        



            ultra_late = (0.995, 1.0)



            # Inject per-image color anchors



            for j in range(n_active):



                pc = per_pixC[j] if j < len(per_pixC) else None



                if pc is not None:



                    cond = node_helpers.conditioning_set_values(cond, {



                        "reference_pixels": [pc],



                        "strength": float(0.12 * beta[j]),



                        "timestep_percent_range": [float(ultra_late[0]), float(ultra_late[1])],



                    }, append=True)



        



            # Tiny ultra-late HF for all images (preserve identity cues symmetrically)



            for j in range(n_active):



                pl = per_pixL[j] if j < len(per_pixL) else None



                if pl is not None:



                    cond = node_helpers.conditioning_set_values(cond, {



                        "reference_pixels": [pl],



                        "strength": float(0.18 * beta[j]),



                        "timestep_percent_range": [0.997, 1.0],



                    }, append=True)





        # Fuse a soft color/style anchor across images (weighted by alpha)



        if n_img > 0:



            per_pixC = []



            for im in images_in:



                if im is None:



                    per_pixC.append(None); continue



                im_bhwc = _bhwc(im)[...,:3]



                lb, _pad = _letterbox(im_bhwc, int(Wc), int(Hc))



                pc = _lowpass_ref(lb, 32)



                pc = _color_lock_to(pc, lb, mix=0.98)



                per_pixC.append(pc)



            # aggregate color anchor



            denom = float(max(1e-6, sum(alpha)))



            for j, pc in enumerate(per_pixC):



                if pc is None: continue



                w = float(0.22 * (alpha[j]/denom))



                cond = node_helpers.conditioning_set_values(cond, {



                    "reference_pixels": [pc],



                    "strength": w,



                    "timestep_percent_range": [float(ultra_late[0]), float(ultra_late[1])],



                }, append=True)

        latent = {"samples": lat,
                  "qi_pad": {"top": int(top), "bottom": int(bottom), "left": int(left), "right": int(right),
                             "orig_h": int(Ht), "orig_w": int(Wt),
                             "compute_h": int(padded.shape[1]), "compute_w": int(padded.shape[2])}}
        return (cond, _ensure_rgb3(_bhwc(image)), latent)

NODE_CLASS_MAPPINGS = {"QI_RefEditEncode_Safe": QI_RefEditEncode_Safe}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_RefEditEncode_Safe": "Qwen一致性编辑编码器 — by wallen0322"}
