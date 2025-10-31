from __future__ import annotations
import torch
import torch.nn.functional as F
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

def _get_geometry_tensor(image_bhwc: torch.Tensor) -> torch.Tensor:
    h, w = int(image_bhwc.shape[1]), int(image_bhwc.shape[2])
    aspect_ratio = w / h
    aspect_tensor = torch.tensor(aspect_ratio, dtype=torch.float32)
    return aspect_tensor

def _get_color_bias_tensor(image_bhwc: torch.Tensor) -> torch.Tensor:
    rgb = _ensure_rgb3(image_bhwc)
    b, c, h, w = rgb.shape
    
    if c == 3:
        mean_rgb = torch.mean(rgb.view(b, c, -1), dim=2)
        std_rgb = torch.std(rgb.view(b, c, -1), dim=2)
        
        color_bias = torch.cat([
            mean_rgb,
            std_rgb
        ], dim=1)
    else:
        color_bias = torch.zeros((b, 2))
        
    return color_bias

class QI_QwenVLClipWrapper:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
        }, "optional": {
            "max_pixels": ("INT", {"default": 1400000, "min": 256000, "max": 1400000}),
            "use_cache": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("vl_conditioning", "prompt_template", "placeholder_tokens")
    FUNCTION = "load"
    CATEGORY = "Qwen2.5"

    def load(self, images: torch.Tensor, max_pixels: int = 1400000, use_cache: bool = True):
        try:
            images_bhwc = _to_bhwc_any(images)
            B, H, W, C = images_bhwc.shape
            
            if W*H > max_pixels:
                scale = (max_pixels / (W*H))**0.5
                new_W = max(8, _round_to(int(W*scale), 8))
                new_H = max(8, _round_to(int(H*scale), 8))
                
                if use_cache:
                    cache_key = f"vl_load_{W}x{H}_{new_W}x{new_H}_{max_pixels}"
                    if cache_key in _compute_cache:
                        cached_result = _compute_cache[cache_key]
                        vl_conditioning = cached_result[0]
                        prompt_template = cached_result[1]
                        placeholder_tokens = cached_result[2]
                    else:
                        resized_images = _resize_bchw_smart(_bchw(images_bhwc), new_W, new_H).movedim(1, -1)
                        
                        geometry_tensor = _get_geometry_tensor(resized_images)
                        color_bias_tensor = _get_color_bias_tensor(resized_images)
                        
                        vl_conditioning = self._create_vl_conditioning(resized_images, geometry_tensor, color_bias_tensor)
                        prompt_template = self._create_prompt_template(B)
                        placeholder_tokens = self._create_placeholder_tokens(B)
                        
                        _compute_cache[cache_key] = (vl_conditioning, prompt_template, placeholder_tokens)
                else:
                    resized_images = _resize_bchw_smart(_bchw(images_bhwc), new_W, new_H).movedim(1, -1)
                    
                    geometry_tensor = _get_geometry_tensor(resized_images)
                    color_bias_tensor = _get_color_bias_tensor(resized_images)
                    
                    vl_conditioning = self._create_vl_conditioning(resized_images, geometry_tensor, color_bias_tensor)
                    prompt_template = self._create_prompt_template(B)
                    placeholder_tokens = self._create_placeholder_tokens(B)
            else:
                geometry_tensor = _get_geometry_tensor(images_bhwc)
                color_bias_tensor = _get_color_bias_tensor(images_bhwc)
                
                if use_cache:
                    cache_key = f"vl_load_{W}x{H}_{max_pixels}"
                    if cache_key in _compute_cache:
                        cached_result = _compute_cache[cache_key]
                        vl_conditioning = cached_result[0]
                        prompt_template = cached_result[1]
                        placeholder_tokens = cached_result[2]
                    else:
                        vl_conditioning = self._create_vl_conditioning(images_bhwc, geometry_tensor, color_bias_tensor)
                        prompt_template = self._create_prompt_template(B)
                        placeholder_tokens = self._create_placeholder_tokens(B)
                        
                        _compute_cache[cache_key] = (vl_conditioning, prompt_template, placeholder_tokens)
                else:
                    vl_conditioning = self._create_vl_conditioning(images_bhwc, geometry_tensor, color_bias_tensor)
                    prompt_template = self._create_prompt_template(B)
                    placeholder_tokens = self._create_placeholder_tokens(B)
            
            return (vl_conditioning, prompt_template, placeholder_tokens)
            
        except Exception as e:
            raise RuntimeError(f"VL loading failed: {e}")

    def _create_vl_conditioning(self, images_bhwc: torch.Tensor, geometry_tensor: torch.Tensor, color_bias_tensor: torch.Tensor):
        vl_conditioning = {
            "images": images_bhwc,
            "geometry": geometry_tensor,
            "color_bias": color_bias_tensor
        }
        return vl_conditioning

    def _create_prompt_template(self, B: int) -> str:
        placeholders = []
        for i in range(B):
            placeholders.append(f"<image_{i}>")
        return " ".join(placeholders)

    def _create_placeholder_tokens(self, B: int) -> str:
        tokens = []
        for i in range(B):
            tokens.append(f"<image_{i}>")
        return " ".join(tokens)
