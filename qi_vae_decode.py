from __future__ import annotations
import torch
import torch.nn.functional as F
import comfy.utils

_ALIGN_M = 32

_compute_cache = {}

def _clear_cache():
    global _compute_cache
    _compute_cache.clear()

def _ceil_to(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m if m > 1 else v

def _floor_to(v: int, m: int) -> int:
    return (v // m) * m if m > 1 else v

def _crop_from_padding(samples_bchw: torch.Tensor, crop_info: dict) -> torch.Tensor:
    top = crop_info.get("top", 0)
    bottom = crop_info.get("bottom", 0)
    left = crop_info.get("left", 0)
    right = crop_info.get("right", 0)
    
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        h, w = int(samples_bchw.shape[2]), int(samples_bchw.shape[3])
        
        if bottom >= h or right >= w:
            return samples_bchw
            
        samples_bchw = samples_bchw[:, :, top:h-bottom, left:w-right]
        
    return samples_bchw

class QI_VAEDecodeLockSize:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "vae": ("VAE",),
            "crop_info": ("STRING", {"default": "0x0+0+0", "multiline": False}),
        }, "optional": {
            "use_cache": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "Qwen2.5"

    def decode(self, samples: dict, vae, crop_info: str = "0x0+0+0", use_cache: bool = True):
        try:
            samples_bchw = samples["samples"]
            h, w = int(samples_bchw.shape[2]), int(samples_bchw.shape[3])
            
            parts = crop_info.split("x")
            if len(parts) != 4:
                raise RuntimeError(f"Invalid crop_info format: {crop_info}")
                
            try:
                top, bottom, left, right = [int(x) for x in parts]
            except ValueError:
                raise RuntimeError(f"Invalid crop_info values: {crop_info}")
            
            if use_cache:
                cache_key = f"vae_decode_{h}x{w}_{crop_info}"
                if cache_key in _compute_cache:
                    decoded_images = _compute_cache[cache_key]
                else:
                    images = vae.decode(samples_bchw)
                    if isinstance(images, dict):
                        decoded_images = images["samples"]
                    else:
                        decoded_images = images
                        
                    decoded_images = _crop_from_padding(decoded_images, {
                        "top": top, "bottom": bottom, "left": left, "right": right
                    })
                    
                    _compute_cache[cache_key] = decoded_images
            else:
                images = vae.decode(samples_bchw)
                if isinstance(images, dict):
                    decoded_images = images["samples"]
                else:
                    decoded_images = images
                    
                decoded_images = _crop_from_padding(decoded_images, {
                    "top": top, "bottom": bottom, "left": left, "right": right
                })
            
            return (decoded_images,)
            
        except Exception as e:
            raise RuntimeError(f"VAE decoding failed: {e}")
