from __future__ import annotations
import torch, torch.nn.functional as F
from typing import Dict, Any, Union, List

def _first_tensor(x) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None: return t
    if isinstance(x, dict):
        # Try common keys first
        for k in ("image", "images", "samples", "data", "result"):
            if k in x:
                t = _first_tensor(x[k])
                if t is not None: return t
        # Otherwise scan all values
        for v in x.values():
            t = _first_tensor(v)
            if t is not None: return t
    return None

def _to_bhwc_img(x: Union[torch.Tensor, Dict, List]) -> torch.Tensor:
    t = _first_tensor(x)
    if t is None:
        raise RuntimeError(f"Unsupported type for image: {type(x)}")

    # detach to avoid grads, ensure contiguous
    t = t.detach()

    # Squeeze leading singleton batch dims if >4D
    while t.ndim > 4 and t.shape[0] == 1:
        t = t.squeeze(0)

    # Handle temporal or latent dims (B,T,C,H,W) -> take first T
    if t.ndim == 5:
        # Heuristic: choose axis with small size (<=4) as channel if exists
        # Assume shape is (B,T,C,H,W) or (B,H,W,C,T)
        # Prefer collapsing "T" by index 1 or -1 if small (<=8)
        if t.shape[1] <= 8:
            t = t[:, 0]
        else:
            t = t[..., 0]

    if t.ndim == 2:
        # H,W -> expand to B,H,W,1 and then to 3 channels
        t = t.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,3)

    elif t.ndim == 3:
        # CHW or HWC
        if t.shape[0] in (1,3,4):   # CHW
            t = t.unsqueeze(0).movedim(1, -1)
        elif t.shape[-1] in (1,3,4):  # HWC
            t = t.unsqueeze(0)
        else:
            # Ambiguous: treat as HWC
            t = t.unsqueeze(0)

    elif t.ndim == 4:
        # BCHW or BHWC or ambiguous
        if t.shape[-1] in (1,3,4):   # BHWC
            pass
        elif t.shape[1] in (1,3,4):  # BCHW
            t = t.movedim(1, -1)
        else:
            # Heuristic: move any 1/3/4-sized axis to channel-last
            guess = None
            for i, d in enumerate(t.shape):
                if d in (1,3,4):
                    guess = i; break
            if guess is not None and guess != 3:
                t = t.movedim(guess, -1)
            else:
                # Fallback: assume BCHW
                t = t.movedim(1, -1)
    else:
        raise RuntimeError(f"Unsupported tensor shape for image: {tuple(t.shape)} (ndim={t.ndim})")

    # Ensure float BHWC in 0..1
    t = t.contiguous().float()
    # some VAE outputs are in [-1,1]
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    t = t.clamp(0,1)

    # If single-channel, expand to RGB
    if t.shape[-1] == 1:
        t = t.repeat(1,1,1,3)

    return t

def _crop_by_qipad(bhwc: torch.Tensor, qi_pad: Dict[str, Any]) -> torch.Tensor:
    H, W = int(bhwc.shape[1]), int(bhwc.shape[2])
    top = int(qi_pad.get("top", 0)); bottom = int(qi_pad.get("bottom", 0))
    left = int(qi_pad.get("left", 0)); right = int(qi_pad.get("right", 0))
    y0 = max(0, min(H, top)); y1 = max(y0, min(H, H - bottom))
    x0 = max(0, min(W, left)); x1 = max(x0, min(W, W - right))
    cropped = bhwc[:, y0:y1, x0:x1, :]
    oh = int(qi_pad.get("orig_h", cropped.shape[1])); ow = int(qi_pad.get("orig_w", cropped.shape[2]))
    dy = cropped.shape[1] - oh; dx = cropped.shape[2] - ow
    if dy != 0 or dx != 0:
        ystart = max(0, (cropped.shape[1] - oh) // 2); xstart = max(0, (cropped.shape[2] - ow) // 2)
        yend = ystart + min(oh, cropped.shape[1]); xend = xstart + min(ow, cropped.shape[2])
        cropped = cropped[:, ystart:yend, xstart:xend, :]
        if cropped.shape[1] != oh or cropped.shape[2] != ow:
            pad_t = max(0, (oh - cropped.shape[1]) // 2); pad_b = oh - cropped.shape[1] - pad_t
            pad_l = max(0, (ow - cropped.shape[2]) // 2); pad_r = ow - cropped.shape[2] - pad_l
            bchw = cropped.movedim(-1, 1)
            bchw = F.pad(bchw, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
            cropped = bchw.movedim(1, -1)
    return cropped

class QI_VAEDecodeLockSize:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "latent": ("LATENT",)},
                "optional": {"force_fp32": ("BOOLEAN", {"default": True}),
                             "move_to_cpu": ("BOOLEAN", {"default": True})}}

    def decode(self, vae, latent, force_fp32: bool = True, move_to_cpu: bool = True):
        x = latent["samples"]
        if isinstance(x, torch.Tensor) and force_fp32 and x.dtype != torch.float32:
            x = x.float()
        with torch.inference_mode():
            img = vae.decode(x)
        bhwc = _to_bhwc_img(img)
        qi_pad = latent.get("qi_pad", {})
        if isinstance(qi_pad, dict) and qi_pad:
            bhwc = _crop_by_qipad(bhwc, qi_pad)
        if move_to_cpu:
            bhwc = bhwc.cpu().contiguous()
        return (bhwc,)

NODE_CLASS_MAPPINGS = {"QI_VAEDecodeLockSize": QI_VAEDecodeLockSize}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_VAEDecodeLockSize": "Qwen VAE 解码（尺寸锁定裁回） — by wallen0322"}
