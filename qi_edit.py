from __future__ import annotations
import torch
import torch.nn.functional as F
import comfy.utils

_ALIGN_M = 32

_compute_cache = {}

def _clear_cache():
    global _compute_cache
    _compute_cache.clear()

def _smoothstep(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    t = (x - a) / max(1e-6, (b - a))
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _get_schedule_mults(num_steps: int, strength: float = 0.6) -> torch.Tensor:
    end_step = int(num_steps * (1.0 - strength))
    
    start_mults = _smoothstep(0.0, 0.25, torch.linspace(0.0, 1.0, num_steps))
    mid_mults = _smoothstep(0.1, 0.9, torch.linspace(0.0, 1.0, num_steps))
    end_mults = _smoothstep(0.75, 1.0, torch.linspace(0.0, 1.0, num_steps))
    
    schedule = torch.ones(num_steps, dtype=torch.float32)
    schedule[:end_step] = start_mults[:end_step] * mid_mults[:end_step]
    schedule[end_step:] = end_mults[end_step:] * mid_mults[end_step:]
    
    return schedule

class QI_TextEncodeQwenImageEdit_Safe:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "text": ("STRING", {"multiline": True, "default": ""}),
            "frontend": ("STRING", {"default": "Qwen2.5-VL", "choices": ["Qwen2.5-VL", "Qwen2.5", "Llama", "ChatGLM"]}),
        }, "optional": {
            "width": ("INT", {"default": 1024, "min": 8, "max": 8192}),
            "height": ("INT", {"default": 1024, "min": 8, "max": 8192}),
            "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            "strength": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0}),
            "use_cache": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Qwen2.5"

    def encode(self, clip, text: str, frontend: str = "Qwen2.5-VL", width: int = 1024, height: int = 1024,
               num_inference_steps: int = 20, strength: float = 0.6, use_cache: bool = True):
        try:
            cache_key = f"text_encode_{frontend}_{width}x{height}_{num_inference_steps}_{strength}"
            
            if use_cache and cache_key in _compute_cache:
                cached_result = _compute_cache[cache_key]
                conditioning = cached_result[0]
            else:
                template = self._get_frontend_template(frontend)
                text_template = f"{template} {text}"
                
                tokens = clip.tokenize(text_template)
                
                width_height = f"{width}x{height}"
                schedule_mults = _get_schedule_mults(num_inference_steps, strength)
                
                result = clip.encode_from_tokens_scheduled(tokens, [width_height], [schedule_mults])
                conditioning = result
                
                if use_cache:
                    _compute_cache[cache_key] = (conditioning,)
            
            return (conditioning,)
            
        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {e}")

    def _get_frontend_template(self, frontend: str) -> str:
        templates = {
            "Qwen2.5-VL": "You are a helpful assistant. Please observe the following image:",
            "Qwen2.5": "Please observe the following image:",
            "Llama": "Please analyze the following image:",
            "ChatGLM": "Please examine the following image:"
        }
        return templates.get(frontend, templates["Qwen2.5-VL"])
