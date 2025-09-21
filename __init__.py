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

NODE_DISPLAY_NAME_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": "Qwen图像编辑编码器 — by wallen0322",
    "QI_RefEditEncode_Safe": "Qwen一致性编辑编码器 — by wallen0322",
    "QI_VAEDecodeLockSize": "Qwen VAE 解码（尺寸锁定裁回） — by wallen0322",
    "QI_QwenVLClipWrapper": "Qwen 2.5 VL 专用加载器（包装） — by wallen0322",
}
