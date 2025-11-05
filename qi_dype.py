"""
DyPE (Dynamic Position Extrapolation) for Qwen-Image
Paper: https://arxiv.org/abs/2510.20766
Reference: https://github.com/wildminder/ComfyUI-DyPE
"""

import torch
import math


def compute_dynamic_scale(timestep, total_steps, base_scale, target_scale, exponent):
    """DyPE动态缩放"""
    t_norm = timestep / total_steps
    kappa_t = target_scale * (t_norm ** exponent)
    scale = base_scale + (kappa_t - base_scale) * (1.0 - t_norm)
    return max(1.0, scale)


class DyPEPatcher:
    def __init__(self, model, width, height, method, enable_dype, exponent):
        self.model = model
        self.width = width
        self.height = height
        self.method = method
        self.enable_dype = enable_dype
        self.exponent = exponent
        self.base_resolution = 1024
        self.scale_factor = max(width, height) / self.base_resolution
        
    def patch_model(self, model):
        m = model.clone()
        if self.scale_factor <= 1.0:
            return m
        
        if not hasattr(m.model, '_dype_forward_orig'):
            m.model._dype_forward_orig = m.model.forward
        
        original_forward = m.model._dype_forward_orig
        patcher = self
        
        def patched_forward(x, timestep, context=None, **kwargs):
            if patcher.enable_dype and timestep is not None:
                t_val = timestep.item() if isinstance(timestep, torch.Tensor) and timestep.numel() == 1 else float(timestep)
                scale = compute_dynamic_scale(t_val, 1000, 1.0, patcher.scale_factor, patcher.exponent)
            else:
                scale = patcher.scale_factor
            
            if hasattr(m.model, 'apply_rope_scaling'):
                m.model.apply_rope_scaling(scale, patcher.method)
            
            return original_forward(x, timestep, context=context, **kwargs)
        
        m.model.forward = patched_forward
        return m


class QI_DyPE:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("MODEL", "INT", "INT")
    RETURN_NAMES = ("model", "width", "height")
    FUNCTION = "apply_dype"
    
    PRESETS = {
        "Custom": (0, 0, 0),
        "1K (1024×1024)": (1024, 1024, 1.0),
        "2K (2048×2048)": (2048, 2048, 1.5),
        "3K (3072×3072)": (3072, 3072, 2.0),
        "4K (4096×4096)": (4096, 4096, 2.0),
        "4K Wide (4096×2304)": (4096, 2304, 2.0),
        "4K Portrait (2304×4096)": (2304, 4096, 2.0),
        "8K (7680×4320)": (7680, 4320, 2.5),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["T2I", "Edit"], {"default": "T2I"}),
                "preset": (list(cls.PRESETS.keys()), {"default": "2K (2048×2048)"}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "method": (["yarn", "ntk", "base"], {"default": "yarn"}),
                "enable_dype": ("BOOLEAN", {"default": True}),
                "dype_exponent": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "preserve_detail": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    def apply_dype(self, model, mode, preset, custom_width, custom_height, 
                   method, enable_dype, dype_exponent, preserve_detail=0.8):
        
        # 解析分辨率
        if preset == "Custom":
            width, height = custom_width, custom_height
            exponent = dype_exponent
        else:
            width, height, exponent = self.PRESETS[preset]
            if dype_exponent != 2.0:  # 用户手动调整了
                exponent = dype_exponent
        
        # Edit模式调整
        if mode == "Edit":
            exponent = exponent * (1.0 + preserve_detail * 0.5)
        
        patcher = DyPEPatcher(model, width, height, method, enable_dype, exponent)
        patched_model = patcher.patch_model(model)
        
        return (patched_model, width, height)


NODE_CLASS_MAPPINGS = {"QI_DyPE": QI_DyPE}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_DyPE": "Qwen Image DyPE — Ultra High Resolution — by wallen0322"}
