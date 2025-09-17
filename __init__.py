from .qi_edit import (
    QI_TextEncodeQwenImageEdit_Safe,
    QI_VAEDecodeHQ,
)
from .qi_refedit import QI_RefEditEncode_Safe

NODE_CLASS_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": QI_TextEncodeQwenImageEdit_Safe,
    "QI_VAEDecodeHQ":                  QI_VAEDecodeHQ,
    "QI_RefEditEncode_Safe":           QI_RefEditEncode_Safe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": "QI • TextEncodeQwenImageEdit — by wallen0322",
    "QI_VAEDecodeHQ":                  "QI • VAE Decode — by wallen0322",
    "QI_RefEditEncode_Safe":           "QI • RefEdit Encode — by wallen0322",
}
