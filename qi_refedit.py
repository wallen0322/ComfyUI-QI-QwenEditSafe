from __future__ import annotations
import torch
import torch.nn.functional as F
import node_helpers
import comfy.utils

_ALIGN_M = 32
_SAFE_MAX_PIX = 3_000_000
_VL_MAX_PIX = 1_400_000

_compute_cache = {}

def _clear_cache():
    global _compute_cache
    _compute_cache.clear()

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
    return t.contiguous()

def _bchw(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] in (1,3,4):
        x = x.movedim(-1, 1)
    return x

def _resize_bchw_smart(x: torch.Tensor, W: int, H: int) -> torch.Tensor:
    x = _bchw(x)
    b, c, h, w = x.shape
    if h==H and w==W:
        return x
    out = comfy.utils.common_upscale(x, W, H, upscale_method="lanczos", crop="center")
    return out

def _ensure_rgb3(x: torch.Tensor) -> torch.Tensor:
    x = _bchw(x)
    b, c, h, w = x.shape
    if c==1:
        return x.repeat(1,3,1,1)
    elif c==4:
        return x[:, :3]
    elif c==3:
        return x
    else:
        raise RuntimeError(f"Unsupported channel count: {c}")

def _letterbox_fit(src_bhwc: torch.Tensor, Wt: int, Ht: int, pad_mode: str="reflect") -> tuple[torch.Tensor, dict]:
    W, H = int(src_bhwc.shape[2]), int(src_bhwc.shape[1])
    Wt = _ceil_to(Wt, 8)
    Ht = _ceil_to(Ht, 8)
    Wt = max(8, min(Wt, _floor_to(_SAFE_MAX_PIX**0.5 * 8, 8)))
    Ht = max(8, min(Ht, _floor_to(_SAFE_MAX_PIX**0.5 * 8, 8)))
    if W==Wt and H==Ht:
        return src_bhwc, dict(top=0,bottom=0,left=0,right=0)
    s = min(Wt/float(W), Ht/float(H))
    Wr = max(1, int(round(W*s)))
    Hr = max(1, int(round(H*s)))
    Wr = _round_to(Wr, 8)
    Hr = _round_to(Hr, 8)
    Wr = max(8, min(Wr, Wt))
    Hr = max(8, min(Hr, Ht))
    
    base = _ensure_rgb3(_resize_bchw_smart(_bchw(src_bhwc), Wr, Hr).movedim(1,-1))
    top = (Ht - Hr)//2
    bottom = Ht - Hr - top
    left = (Wt - Wr)//2
    right = Wt - Wr - left
    
    top = (top // 2) * 2
    bottom = Ht - Hr - top
    left = (left // 2) * 2
    right = Wt - Wr - left
    
    bchw = _bchw(base)
    if top>0 or bottom>0 or left>0 or right>0:
        h, w = int(bchw.shape[2]), int(bchw.shape[3])
        mode = pad_mode
        if h < 2 or w < 2 or left >= w or right >= w or top >= h or bottom >= h:
            mode = "replicate"
        bchw = F.pad(bchw, (left,right,top,bottom), mode=mode)
    return _ensure_rgb3(bchw.movedim(1,-1)), dict(top=int(top),bottom=int(bottom),left=int(left),right=int(right))

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
    return l11, l21, l22, inv_l11, inv_l22

def _make_ref_images(base_bhwc: torch.Tensor, target_size: tuple[int,int], batch_idxs: list[int] = None):
    target_H, target_W = target_size
    H, W = int(base_bhwc.shape[1]), int(base_bhwc.shape[2])
    
    if batch_idxs is None:
        batch_idxs = list(range(base_bhwc.shape[0]))
    
    ref_images = {}
    
    if H>0 and W>0 and target_H>0 and target_W>0:
        s_low = min(target_W/W, target_H/H) * 0.25
        s_base = min(target_W/W, target_H/H)
        s_high = min(target_W/W, target_H/H) * 2.0
        
        for i in batch_idxs:
            low_bchw = _resize_bchw_smart(_bchw(base_bhwc[i:i+1]), max(1, int(W*s_low)), max(1, int(H*s_low)))
            base_bchw = _resize_bchw_smart(_bchw(base_bhwc[i:i+1]), max(1, int(W*s_base)), max(1, int(H*s_base)))
            high_bchw = _resize_bchw_smart(_bchw(base_bhwc[i:i+1]), max(1, int(W*s_high)), max(1, int(H*s_high)))
            
            low = _ensure_rgb3(low_bchw).movedim(1, -1)[0]
            base = _ensure_rgb3(base_bchw).movedim(1, -1)[0]
            high = _ensure_rgb3(high_bchw).movedim(1, -1)[0]
            
            ref_images[f"{i}_low"] = low
            ref_images[f"{i}_base"] = base
            ref_images[f"{i}_high"] = high
            
    return ref_images

def _to_ycbcr(t: torch.Tensor) -> torch.Tensor:
    t = _bchw(t)
    r, g, b = t[:,0], t[:,1], t[:,2]
    y = 0.299*r + 0.587*g + 0.114*b
    cb = (b - y)*0.564
    cr = (r - y)*0.713
    return torch.stack([y, cb, cr], dim=1)

def _from_ycbcr(y_cb_cr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = y_cb_cr[:,0], y_cb_cr[:,1], y_cb_cr[:,2]
    r = y + 1.403*cr
    g = y - 0.344*cb - 0.714*cr
    b = y + 1.773*cb
    return torch.stack([r, g, b], dim=1)

def _clip_chroma_stats_ycbcr(images: torch.Tensor) -> torch.Tensor:
    ycbcr = _to_ycbcr(_ensure_rgb3(images))
    cb, cr = ycbcr[:,1], ycbcr[:,2]
    mean_cb = torch.mean(cb, dim=[2,3], keepdim=True)
    mean_cr = torch.mean(cr, dim=[2,3], keepdim=True)
    var_cb = torch.var(cb, dim=[2,3], keepdim=True)
    var_cr = torch.var(cr, dim=[2,3], keepdim=True)
    cov_cbcr = torch.mean((cb-mean_cb)*(cr-mean_cr), dim=[2,3], keepdim=True)
    
    eps = 1e-6
    a = torch.clamp(var_cb, min=eps)
    b = cov_cbcr
    c = torch.clamp(var_cr, min=eps)
    
    l11, l21, l22, inv_l11, inv_l22 = _chol2x2(a, b, c, eps)
    
    def transform(cb_t, cr_t):
        cb_adj = cb_t - mean_cb
        cr_adj = cr_t - mean_cr
        l21_cb = l21*cb_adj
        temp = cr_adj - l21_cb
        l11_cb = l11*cb_adj
        l22_temp = l22*temp
        out_cb = inv_l11*l11_cb
        out_cr = inv_l22*l22_temp
        return out_cb + mean_cb, out_cr + mean_cr
        
    return transform

def _prepare_reference_conditioning(reference_images: dict, images_bhwc: torch.Tensor, conditioning_timesteps: torch.Tensor, true_cfg_scale: float = 1.0) -> dict:
    ref_count = len(reference_images) // 3
    ref_cond = dict()
    
    for i in range(ref_count):
        low = reference_images[f"{i}_low"]
        base = reference_images[f"{i}_base"] 
        high = reference_images[f"{i}_high"]
        
        low_cond = node_helpers.conditioning_set_values(images_bhwc, conditioning_timesteps, 
                                                       {"low_res_ref": low, "ref_idx": i})
        base_cond = node_helpers.conditioning_set_values(images_bhwc, conditioning_timesteps,
                                                         {"ref_res_base": base, "ref_idx": i})
        high_cond = node_helpers.conditioning_set_values(images_bhwc, conditioning_timesteps,
                                                         {"high_res_ref": high, "ref_idx": i})
        
        ref_cond[f"{i}"] = [low_cond, base_cond, high_cond]
    
    if "chromalock_transform" in reference_images:
        transform = reference_images["chromalock_transform"]
        ref_cond["chromalock"] = transform
        
    if true_cfg_scale > 1.0:
        ref_cond["true_cfg_scale"] = true_cfg_scale
        
    return ref_cond

class QI_RefEditEncode_Safe:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "target_resolution": ("STRING", {"default": "1024x1024", "multiline": False}),
        }, "optional": {
            "reference_image": ("IMAGE",),
            "reference_resolution": ("STRING", {"default": "512x512", "multiline": False}),
            "color_lock": ("BOOLEAN", {"default": True}),
            "use_cache": ("BOOLEAN", {"default": True}),
            "true_cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0}),
        }}
    
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("ref_conditioning", "prompt_template", "placeholder_tokens", "image_shapes")
    FUNCTION = "encode"
    CATEGORY = "Qwen2.5"

    def encode(self, images: torch.Tensor, target_resolution: str, reference_image: torch.Tensor=None, 
               reference_resolution: str="512x512", color_lock: bool=True, use_cache: bool=True, true_cfg_scale: float=3.0):
        try:
            Wt, Ht = [int(x) for x in target_resolution.split("x")]
            Wr, Hr = [int(x) for x in reference_resolution.split("x")]
            
            images_bhwc = _to_bhwc_any(images)
            base_H, base_W = images_bhwc.shape[1], images_bhwc.shape[2]
            B = images_bhwc.shape[0]
            
            if Wt*Ht > _SAFE_MAX_PIX or Wr*Hr > _SAFE_MAX_PIX:
                raise RuntimeError(f"Resolution too large: target {Wt}x{Ht}, ref {Wr}x{Hr}")
            
            if use_cache:
                cache_key = f"{base_W}x{base_H}_{Wt}x{Ht}_{Wr}x{Hr}_{color_lock}_{true_cfg_scale}"
                if cache_key in _compute_cache:
                    cached_result = _compute_cache[cache_key]
                    prompt_template = cached_result[1]
                    placeholder_tokens = cached_result[2]
                    image_shapes = cached_result[3]
                else:
                    ref_images = self._prepare_reference_images(images_bhwc, Wr, Hr, color_lock, use_cache)
                    ref_cond = self._create_conditioning(ref_images, true_cfg_scale)
                    prompt_template = self._create_prompt_template(B)
                    placeholder_tokens = self._create_placeholder_tokens(B)
                    image_shapes = f"{base_W}x{base_H}"
                    
                    _compute_cache[cache_key] = (ref_cond, prompt_template, placeholder_tokens, image_shapes)
            else:
                ref_images = self._prepare_reference_images(images_bhwc, Wr, Hr, color_lock, use_cache)
                ref_cond = self._create_conditioning(ref_images, true_cfg_scale)
                prompt_template = self._create_prompt_template(B)
                placeholder_tokens = self._create_placeholder_tokens(B)
                image_shapes = f"{base_W}x{base_H}"
            
            return (ref_cond, prompt_template, placeholder_tokens, image_shapes)
            
        except Exception as e:
            raise RuntimeError(f"Reference encoding failed: {e}")

    def _prepare_reference_images(self, images_bhwc, Wr, Hr, color_lock, use_cache):
        ref_images = {}
        B = images_bhwc.shape[0]
        target_size = (Hr, Wr)
        
        cache_key = f"ref_images_{images_bhwc.shape}_{Hr}x{Wr}_{color_lock}"
        if use_cache and cache_key in _compute_cache:
            ref_images = _compute_cache[cache_key]
        else:
            if color_lock:
                clip_transform = _clip_chroma_stats_ycbcr(images_bhwc)
                ref_images["chromalock_transform"] = clip_transform
                
            ref_images.update(_make_ref_images(images_bhwc, target_size))
            
            if use_cache:
                _compute_cache[cache_key] = ref_images
                
        return ref_images

    def _create_conditioning(self, ref_images, true_cfg_scale):
        B = len(ref_images) // 3 if "chromalock_transform" not in ref_images else (len(ref_images) - 1) // 3
        conditioning_timesteps = torch.ones(B, dtype=torch.float32)
        
        ref_cond = _prepare_reference_conditioning(ref_images, ref_images.get("ref_images", {}), conditioning_timesteps, true_cfg_scale)
        
        return ref_cond

    def _create_prompt_template(self, B):
        placeholders = []
        for i in range(B):
            placeholders.append(f"<image_{i}>")
        
        prompt_template = " ".join(placeholders)
        return prompt_template

    def _create_placeholder_tokens(self, B):
        tokens = []
        for i in range(B):
            tokens.append(f"<image_{i}>")
        return " ".join(tokens)
