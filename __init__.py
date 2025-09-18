from .qi_edit import (
    QI_TextEncodeQwenImageEdit_Safe,
    QI_VAEDecodeHQ,
)
from .qi_refedit import QI_RefEditEncode_Safe

NODE_CLASS_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": QI_TextEncodeQwenImageEdit_Safe,
    "QI_VAEDecodeHQ": QI_VAEDecodeHQ,
    "QI_RefEditEncode_Safe": QI_RefEditEncode_Safe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": "Qwen图像编辑编码器 — by wallen0322",
    "QI_VAEDecodeHQ": "Qwen高质量VAE解码器 — by wallen0322",
    "QI_RefEditEncode_Safe": "Qwen一致性编辑编码器 — by wallen0322",
}
