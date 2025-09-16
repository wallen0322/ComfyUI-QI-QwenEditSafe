from .qi_edit import (
    QI_TextEncodeQwenImageEdit_Safe,
    QI_TextEncodeQwenImageEdit_CN,
    QI_VAEDecodeHQ,
    QI_VAEDecodeHQ_CN,
)

NODE_CLASS_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": QI_TextEncodeQwenImageEdit_Safe,
    "QI_TextEncodeQwenImageEdit_CN":   QI_TextEncodeQwenImageEdit_CN,
    "QI_VAEDecodeHQ":                  QI_VAEDecodeHQ,
    "QI_VAEDecodeHQ_CN":               QI_VAEDecodeHQ_CN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": "QI • TextEncodeQwenImageEdit — by wallen0322",
    "QI_TextEncodeQwenImageEdit_CN":   "QI • 文生图编辑 — by wallen0322",
    "QI_VAEDecodeHQ":                  "QI • VAE Decode — by wallen0322",
    "QI_VAEDecodeHQ_CN":               "QI • VAE 解码 — by wallen0322",
}
