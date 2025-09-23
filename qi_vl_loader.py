from __future__ import annotations
import torch
import torch.nn.functional as F
import comfy.utils

# ---------- helpers (IMAGE: BHWC in [0,1]) ----------
def _to_bhwc(img):
    t = img
    if isinstance(t, (list, tuple)):
        t = t[0]
    if not isinstance(t, torch.Tensor):
        raise RuntimeError("Invalid IMAGE")
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(-1)
    elif t.ndim == 3:
        if t.shape[0] in (1,3,4):
            t = t.unsqueeze(0).movedim(1,-1)
        else:
            t = t.unsqueeze(0)
    elif t.ndim == 4:
        if t.shape[-1] in (1,3,4): pass
        elif t.shape[1] in (1,3,4): t = t.movedim(1,-1)
        else: t = t.movedim(1,-1)
    else:
        raise RuntimeError("Unsupported IMAGE shape")
    t = t.float()
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    if t.shape[-1] == 1:
        t = t.repeat(1,1,1,3)
    return t.clamp(0,1).contiguous()

def _bchw(img_bhwc): return _to_bhwc(img_bhwc).movedim(-1,1).contiguous()
def _bhwc_from_bchw(x): return x.movedim(1,-1).contiguous()

def _center_crop_to_square(bchw):
    _,_,H,W = bchw.shape
    if H==W: return bchw
    if H > W:
        top = (H-W)//2; return bchw[:,:,top:top+W,:]
    else:
        left = (W-H)//2; return bchw[:,:,:,left:left+H]

def _resize(bchw, w, h):
    return comfy.utils.common_upscale(bchw, w, h, "bicubic", "disabled")

def _desaturate(bchw, amt=0.12):
    if amt <= 1e-6: return bchw
    y = bchw[:,0:1]*0.299 + bchw[:,1:2]*0.587 + bchw[:,2:3]*0.114
    gray = torch.cat([y,y,y], dim=1)
    return (1.0-amt)*bchw + amt*gray

def _tiers_area(H, W):
    area = H*W
    tiers = [int(0.9e6), int(1.2e6), int(1.4e6)]
    for t in tiers:
        if area <= t: return H, W
    s = (tiers[-1]/float(area))**0.5
    return max(1,int(H*s)), max(1,int(W*s))

# ---------- CLIP proxy that preprocesses VL image ----------
class _CLIPProxyQwenVL:
    def __init__(self, base_clip, fixed_size=672, geometry="letterbox",
                 neutralize=True, desat=0.12, tiers_mode="off", color_mode="grayscale"):
        self.base = base_clip
        self.fixed_size = int(fixed_size)
        self.geometry = geometry  # "center_crop" or "letterbox"
        self.neutralize = bool(neutralize)
        self.desat = float(desat)
        self.tiers_mode = tiers_mode  # "tiers" | "limit" | "off"
        self.color_mode = color_mode  # "original" | "neutral_gray" | "grayscale"

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _prep_single(self, img):
        bhwc = _to_bhwc(img)
        bchw = _bchw(bhwc)
        _,_,H,W = bchw.shape

        if self.tiers_mode == "tiers":
            Ht,Wt = _tiers_area(H,W)
        elif self.tiers_mode == "limit":
            maxpix = int(1.4e6)
            area = H*W
            if area > maxpix:
                s = (maxpix/float(area))**0.5
                Ht,Wt = max(1,int(H*s)), max(1,int(W*s))
            else:
                Ht,Wt = H,W
        else:
            Ht,Wt = H,W

        if self.geometry == "center_crop":
            sq = _center_crop_to_square(bchw)
            bchw2 = _resize(sq, self.fixed_size, self.fixed_size)
        else:
            s = min(self.fixed_size/Wt, self.fixed_size/Ht)
            Wr = max(1,int(round(Wt*s))); Hr = max(1,int(round(Ht*s)))
            base = _resize(bchw, Wr, Hr)
            pad_l = (self.fixed_size - Wr)//2
            pad_r = self.fixed_size - Wr - pad_l
            pad_t = (self.fixed_size - Hr)//2
            pad_b = self.fixed_size - Hr - pad_t
            bchw2 = torch.nn.functional.pad(base, (pad_l,pad_r,pad_t,pad_b),
                                            mode="constant", value=0.5)  # neutral 50% gray

        # color handling for VL
        if self.color_mode == "grayscale":
            y = bchw2[:,0:1]*0.299 + bchw2[:,1:2]*0.587 + bchw2[:,2:3]*0.114
            bchw2 = torch.cat([y,y,y], dim=1)
        elif self.color_mode == "neutral_gray":
            mu = bchw2.mean(dim=(2,3), keepdim=True)
            bchw2 = (bchw2 - mu) * 0.0 + mu

        if self.neutralize and self.desat > 0 and self.color_mode == "original":
            bchw2 = _desaturate(bchw2, self.desat)

        return _bhwc_from_bchw(bchw2).clamp(0,1)

    def tokenize(self, prompt, images=None, **kwargs):
        imgs = images
        if images is not None and len(images)>0:
            proc = [self._prep_single(im) for im in images]
            imgs = proc
        return self.base.tokenize(prompt, images=imgs, **kwargs)

class QI_QwenVLClipWrapper:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "wrap"
    RETURN_NAMES = ("clip",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "fixed_size": ("INT", {"default": 672, "min": 224, "max": 1024, "step": 32}),
            "geometry": (["letterbox","center_crop"], {"default": "letterbox"}),
            "color_mode": (["grayscale","neutral_gray","original"], {"default": "grayscale"}),
            "neutralize": ("BOOLEAN", {"default": True}),
            "desaturate": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
            "mp_policy": (["off","limit","tiers"], {"default": "off"}),
        }}

    def wrap(self, clip, fixed_size=672, geometry="letterbox", color_mode="grayscale",
             neutralize=True, desaturate=0.12, mp_policy="off"):
        proxy = _CLIPProxyQwenVL(clip, fixed_size=fixed_size, geometry=geometry,
                                 neutralize=neutralize, desat=desaturate,
                                 tiers_mode=mp_policy, color_mode=color_mode)
        return (proxy,)

NODE_CLASS_MAPPINGS = {"QI_QwenVLClipWrapper": QI_QwenVLClipWrapper}
NODE_DISPLAY_NAME_MAPPINGS = {"QI_QwenVLClipWrapper": "Qwen 2.5 VL 专用加载器（包装） — by wallen0322"}
