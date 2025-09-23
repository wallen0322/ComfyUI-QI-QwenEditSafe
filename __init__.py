from .qi_edit import QI_TextEncodeQwenImageEdit_Safe
from .qi_refedit import QI_RefEditEncode_Safe
from .qi_vae_decode import QI_VAEDecodeLockSize
from .qi_vl_loader import QI_QwenVLClipWrapper

NODE_CLASS_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": QI_TextEncodeQwenImageEdit_Safe,
    "QI_RefEditEncode_Safe": QI_RefEditEncode_Safe,
    "QI_VAEDecodeLockSize": QI_VAEDecodeLockSize,
    "QI_QwenVLClipWrapper": QI_QwenVLClipWrapper,
}

# Default (fallback) English display names. Your locale files can override these.
NODE_DISPLAY_NAME_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": "Qwen Image Edit — Text Encoder (Safe) — by wallen0322",
    "QI_RefEditEncode_Safe": "Qwen Consistency Edit — Encoder — by wallen0322",
    "QI_VAEDecodeLockSize": "Qwen VAE Decode (Lock Size & Crop Back) — by wallen0322",
    "QI_QwenVLClipWrapper": "Qwen 2.5 VL Loader (Wrapper) — by wallen0322",
}
