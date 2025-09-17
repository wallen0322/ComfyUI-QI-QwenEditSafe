from .qi_edit import (
    QI_TextEncodeQwenImageEdit_Safe,
    QI_VAEDecodeHQ,
)

NODE_CLASS_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": QI_TextEncodeQwenImageEdit_Safe,
    "QI_VAEDecodeHQ":                  QI_VAEDecodeHQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": "QI • TextEncodeQwenImageEdit — by wallen0322",
    "QI_VAEDecodeHQ":                  "QI • VAE Decode — by wallen0322",
}
