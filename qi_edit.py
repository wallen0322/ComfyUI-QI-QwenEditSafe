from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch, torch.nn.functional as F
import node_helpers, comfy.utils

# ----------------- tensor utils -----------------
def _to_bhwc_any(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[0] in (1,3,4): x = x.unsqueeze(0).movedim(1,-1)
        elif x.shape[-1] in (1,3,4): x = x.unsqueeze(0)
        else: x = x.unsqueeze(0).movedim(1,-1)
    elif x.ndim == 4:
        if x.shape[-1] in (1,3,4): pass
        elif x.shape[1] in (1,3,4): x = x.movedim(1,-1)
        else: x = x.movedim(1,-1)
    elif x.ndim == 5:
        if x.shape[2] in (1,3,4):
            B,T,C,H,W = x.shape; x = x.reshape(B*T, C, H, W).movedim(1,-1)
        elif x.shape[-1] in (1,3,4):
            B,T,H,W,C = x.shape; x = x.reshape(B*T, H, W, C)
        else:
            B,C,H,W,T = x.shape; x = x.permute(0,4,1,2,3).reshape(B*T,C,H,W).movedim(1,-1)
    else:
        raise RuntimeError(f"Unsupported ndim {x.ndim}")
    return x.float().clamp(0,1).contiguous()

def _ensure_rgb3(bhwc: torch.Tensor) -> torch.Tensor:
    C = int(bhwc.shape[-1])
    if C==3: return bhwc
    if C==1: return bhwc.repeat(1,1,1,3)
    return bhwc[...,:3]

def _bhwc(x: torch.Tensor) -> torch.Tensor:
    return _ensure_rgb3(_to_bhwc_any(x))

def _bchw(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-1,1).contiguous()

def _ceil_to(v:int, m:int)->int: return ((v+m-1)//m)*m if m>1 else v

def _clampf(x: float, lo: float, hi: float) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

def _choose_method(src_w:int, src_h:int, dst_w:int, dst_h:int)->str:
    if dst_w*dst_h <= 0: return "area"
    if dst_w >= src_w or dst_h >= src_h: return "lanczos"
    return "area"

def _resize_bchw(x:torch.Tensor,Wt:int,Ht:int,method:str)->torch.Tensor:
    return comfy.utils.common_upscale(x,Wt,Ht,method,"disabled")

def _resize_bchw_smart(x:torch.Tensor,Wt:int,Ht:int)->torch.Tensor:
    _,_,H,W = x.shape
    method = _choose_method(W,H,Wt,Ht)
    return _resize_bchw(x, Wt, Ht, method)

def _box_blur3(bchw: torch.Tensor, k:int=3)->torch.Tensor:
    if k<=1: return bchw
    if k%2==0: k+=1
    pad=k//2; C=bchw.shape[1]
    ker=torch.ones((C,1,k,k),dtype=bchw.dtype,device=bchw.device)/(k*k)
    x=F.pad(bchw,(pad,pad,pad,pad),mode="reflect")
    return F.conv2d(x, ker, groups=C)

def _lowpass_ref(bhwc: torch.Tensor, size:int=64)->torch.Tensor:
    bchw=_bchw(bhwc)
    bh=_resize_bchw(bchw, size, size, "area")
    bh=_resize_bchw_smart(bh, bchw.shape[-1], bchw.shape[-2])
    return _ensure_rgb3(bh.movedim(1,-1)).clamp(0,1)

def _hf_ref(bhwc: torch.Tensor, alpha: float=0.4, blur_k:int=3) -> torch.Tensor:
    bchw=_bchw(bhwc)
    blur=_box_blur3(bchw, blur_k)
    detail=(bchw - blur)
    out=(bchw + alpha*detail).clamp(0,1)
    return _ensure_rgb3(out.movedim(1,-1))

def _limit_area_keep_ar_smart(bhwc: torch.Tensor, max_pixels: int, align_multiple:int=8) -> torch.Tensor:
    if max_pixels is None or max_pixels <= 0: return bhwc
    B,H,W,C = bhwc.shape
    area = H*W
    if area <= max_pixels: return bhwc
    scale = (max_pixels/float(area))**0.5
    Ht, Wt = max(1,int(H*scale)), max(1,int(W*scale))
    Ht, Wt = _ceil_to(Ht, align_multiple), _ceil_to(Wt, align_multiple)
    bchw = _bchw(bhwc)
    bchw = _resize_bchw_smart(bchw, Wt, Ht)
    return _ensure_rgb3(bchw.movedim(1,-1)).contiguous()

def _rgb_stats(bhwc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bchw=_bchw(bhwc)
    mean=bchw.mean(dim=(2,3), keepdim=True)
    std=bchw.std(dim=(2,3), keepdim=True).clamp_min(1e-6)
    return mean, std

def _match_color(img_bhwc: torch.Tensor, ref_mean: torch.Tensor, ref_std: torch.Tensor) -> torch.Tensor:
    x=_bchw(img_bhwc)
    mean=x.mean(dim=(2,3), keepdim=True); std=x.std(dim=(2,3), keepdim=True).clamp_min(1e-6)
    x=(x-mean)/std*ref_std + ref_mean
    return _ensure_rgb3(x.movedim(1,-1)).clamp(0,1)

# ----------------- core nodes -----------------
class QI_TextEncodeQwenImageEdit_Safe:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("CONDITIONING","IMAGE","LATENT")
    RETURN_NAMES = ("conditioning","image","latent")
    FUNCTION = "encode"

    _DEF_PIXELS_SOURCE = "recon"   # hidden
    _DEF_LATENT_LOWPASS = 0        # hidden
    _DEF_LATENT_W      = 1.00      # hidden
    _DEF_PIXEL_W       = 0.50      # hidden
    _ALIGN_MULTIPLE    = 8         # hidden
    _SMART_RESAMPLE    = True      # hidden

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
                    "clip":("CLIP",),
                    "prompt":("STRING",{"multiline":True,"default":""}),
                    "image":("IMAGE",),
                    "vae":("VAE",),
                },
                "optional":{
                    "no_resize_pad":("BOOLEAN",{"default":True}),
                    "pad_mode":(["reflect","replicate"],{"default":"reflect"}),
                    "grid_multiple":("INT",{"default":64,"min":8,"max":128,"step":8}),
                    "inject_mode":(["both","latents","pixels"],{"default":"both"}),
                    "encode_fp32":("BOOLEAN",{"default":True}),
                    "prompt_emphasis":("FLOAT",{"default":0.60,"min":0.0,"max":1.0,"step":0.05}),
                    "vl_max_pixels":("INT",{"default":16_777_216,"min":0,"max":16_777_216,"step":65536}),
                }}

    # --- adaptive schedule (multi-scale pixel anchoring) ---
    def _derive_schedule(self, emph: float, prompt: str, H:int, W:int) -> Dict[str, Any]:
        plen = len(prompt.strip())
        complexity = _clampf(plen / 120.0, 0.0, 1.0)  # 0~1
        area_norm = _clampf((H*W) / (1024*1024), 0.3, 2.0)  # normalize by 1MP

        base_lat_w = self._DEF_LATENT_W
        base_pix_w = self._DEF_PIXEL_W

        lat_w = _clampf(base_lat_w * (1.0 + 0.6*(emph-0.5)) * (1.0 + 0.10*complexity), 0.85, 1.50)
        pix_w = _clampf(base_pix_w * (1.0 - 0.4*(emph-0.5)) * (1.0 + 0.10*area_norm), 0.35, 1.20)

        # geometry lock: longer early latent, guaranteed early pixel
        lat_end   = 0.36 if emph >= 0.7 else 0.40
        pix_early = [0.30, 0.58]    # low-pass pixel
        pix_mid   = [0.45, 0.75]    # base pixel
        pix_late  = [0.82, 1.00]    # HF pixel

        # strengths (auto-tune with emph & complexity)
        w_lat   = lat_w
        w_pix_m = max(0.55, pix_w) if emph>=0.7 else pix_w
        w_pix_e = _clampf(0.35 + 0.30*emph + 0.15*complexity, 0.30, 0.80)
        w_pix_l = _clampf(0.18 + 0.20*emph, 0.15, 0.45)

        # repetitions
        lat_rep = 1 + (1 if emph>=0.6 else 0)
        pix_rep = 1  # at least one

        return {
            "lat_range":[0.0, float(lat_end)], "lat_w":float(w_lat), "lat_rep":int(lat_rep),
            "pixE_range": pix_early, "pixE_w": float(w_pix_e),
            "pixM_range": pix_mid,   "pixM_w": float(w_pix_m), "pix_rep":int(pix_rep),
            "pixL_range": pix_late,  "pixL_w": float(w_pix_l),
        }

    def _derive_schedule(self, prompt_emphasis, prompt=None, H=None, W=None):
            e = max(0.0, min(1.0, float(prompt_emphasis)))
            lat_range  = [0.00, 0.36]
            pixE_range = [0.30, 0.52]  # LF
            pixM_range = [0.52, 0.78]  # MF
            pixL_range = [0.86, 1.00]  # HF
        
            lat_w  = max(0.90, min(1.45, 0.90 + 0.55*e))
            pixE_w = max(0.20, min(0.40, 0.40 - 0.20*e))
            pixM_w = max(0.40, min(0.70, 0.70 - 0.30*e))
            pixL_w = max(0.12, min(0.30, 0.12 + 0.18*e))
        
            lat_rep = 2 if e >= 0.70 else 1
            pix_rep = 1
        
            # Backward-compat aliases (if caller expects a single pixel band):
            pix_range = pixM_range
            pix_w = pixM_w
        
            return {
                "lat_range": lat_range, "lat_w": float(lat_w), "lat_rep": int(lat_rep),
                "pixE_range": pixE_range, "pixE_w": float(pixE_w),
                "pixM_range": pixM_range, "pixM_w": float(pixM_w),
                "pixL_range": pixL_range, "pixL_w": float(pixL_w),
                "pix_rep": int(pix_rep),
                # aliases:
                "pix_range": pix_range, "pix_w": float(pix_w),
            }
    def encode(self, clip, prompt, image, vae,
               no_resize_pad=True, pad_mode="reflect", grid_multiple=64,
               inject_mode="both", encode_fp32=True, prompt_emphasis=0.60,
               vl_max_pixels=16_777_216):

        src=_bhwc(image)[...,:3]
        src_mean, src_std = _rgb_stats(src)
        H,W=src.shape[1],src.shape[2]

        gm = max(8, int(grid_multiple))
        if gm % self._ALIGN_MULTIPLE != 0:
            gm = _ceil_to(gm, self._ALIGN_MULTIPLE)

        Ht,Wt=_ceil_to(H,gm),_ceil_to(W,gm)
        bchw=_bchw(src); top=left=bottom=right=0
        if no_resize_pad:
            ph,pw=Ht-H,Wt-W
            if ph>0 or pw>0:
                top=ph//2; bottom=ph-top; left=pw//2; right=pw-left
                bchw=F.pad(bchw,(left,right,top,bottom),mode=pad_mode)
        padded=_ensure_rgb3(bchw.movedim(1,-1)).clamp(0,1)

        vl_img = _limit_area_keep_ar_smart(padded, int(vl_max_pixels) if vl_max_pixels is not None else 0, self._ALIGN_MULTIPLE) if self._SMART_RESAMPLE else padded

        tokens=clip.tokenize(prompt, images=[vl_img])
        cond=clip.encode_from_tokens_scheduled(tokens)

        # VAE encode + optional recon (AMP only when CUDA)
        lat=vae.encode(padded)
        if encode_fp32 and isinstance(lat,torch.Tensor) and lat.dtype!=torch.float32:
            lat=lat.float()

        need_pixels = inject_mode in ("both","pixels")
        need_recon  = need_pixels and (self._DEF_PIXELS_SOURCE=="recon")

        recon_bhwc: Optional[torch.Tensor] = None
        if need_recon:
            try:
                dev = lat.device if isinstance(lat, torch.Tensor) else torch.device("cpu")
                use_cuda_amp = (isinstance(dev, torch.device) and dev.type=="cuda")
            except:
                use_cuda_amp=False
            if use_cuda_amp:
                with torch.cuda.amp.autocast(True, dtype=torch.float16):
                    recon = vae.decode(lat)
            else:
                with torch.inference_mode():
                    recon = vae.decode(lat)
            recon_bhwc = _bhwc(recon)

        # multi-scale refs
        pix_base = recon_bhwc if (need_recon and recon_bhwc is not None) else padded
        pixE = _lowpass_ref(pix_base, 64) if need_pixels else None
        pixM = pix_base if need_pixels else None
        pixL = _hf_ref(pix_base, alpha=0.4, blur_k=3) if need_pixels else None

        sch = self._derive_schedule(float(prompt_emphasis), prompt, H, W)

        # latent anchors
        if inject_mode in ("both","latents"):
            for _ in range(sch["lat_rep"]):
                cond=node_helpers.conditioning_set_values(
                    cond, {"reference_latents":[lat],
                           "strength": sch["lat_w"],
                           "timestep_percent_range": sch["lat_range"]},
                    append=True)

        # pixel anchors (E/M/L)
        if need_pixels:
            if pixE is not None:
                cond=node_helpers.conditioning_set_values(
                    cond, {"reference_pixels":[pixE],
                           "strength": sch["pixE_w"],
                           "timestep_percent_range": sch["pixE_range"]},
                    append=True)
            if pixM is not None:
                for _ in range(sch["pix_rep"]):
                    cond=node_helpers.conditioning_set_values(
                        cond, {"reference_pixels":[pixM],
                               "strength": sch["pixM_w"],
                               "timestep_percent_range": sch["pixM_range"]},
                        append=True)
            if pixL is not None:
                cond=node_helpers.conditioning_set_values(
                    cond, {"reference_pixels":[pixL],
                           "strength": sch["pixL_w"],
                           "timestep_percent_range": sch["pixL_range"]},
                    append=True)

        latent={"samples":lat,
                "qi_pad":{"top":int(top),"bottom":int(bottom),"left":int(left),"right":int(right),
                          "orig_h":int(H),"orig_w":int(W)},
                "qi_color":{"mean":src_mean.cpu().numpy().tolist(), "std":src_std.cpu().numpy().tolist()}}
        return (cond, src, latent)

# ----------------- VAE decode (unchanged API) -----------------
class QI_VAEDecodeHQ:
    CATEGORY="QI by wallen0322"
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("image",); FUNCTION="decode"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"vae":("VAE",),"latent":("LATENT",)},
                "optional":{"force_fp32":("BOOLEAN",{"default":True}),
                            "preserve_color":("BOOLEAN",{"default":True}),
                            "unsharp_amount":("FLOAT",{"default":0.0,"min":0.0,"max":1.0,"step":0.05}),
                            "move_to_cpu":("BOOLEAN",{"default":True}),}}
    def _unsharp(self, bchw: torch.Tensor, amt: float)->torch.Tensor:
        if amt<=0: return bchw
        k=torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=bchw.dtype,device=bchw.device).view(1,1,3,3)
        C=bchw.shape[1]; k=k.repeat(C,1,1,1)
        pad=1; x=F.pad(bchw,(pad,pad,pad,pad),mode="reflect")
        return (1-amt)*bchw + amt*F.conv2d(x,k,groups=C)
    def decode(self, vae, latent, force_fp32=True, preserve_color=True, unsharp_amount=0.0, move_to_cpu=True):
        x=latent["samples"]
        if force_fp32 and isinstance(x,torch.Tensor) and x.dtype!=torch.float32: x=x.float()
        with torch.inference_mode():
            img=vae.decode(x)
        if isinstance(img,(list,tuple)): img=img[0]
        bchw=_bchw(img).movedim(-1,1).contiguous()
        if float(unsharp_amount)>0:
            bchw=self._unsharp(bchw, float(unsharp_amount))
        bhwc=_ensure_rgb3(bchw.movedim(1,-1)).clamp(0,1)
        meta=latent.get("qi_pad",None)
        if meta:
            top,bottom,left,right=int(meta.get("top",0)),int(meta.get("bottom",0)),int(meta.get("left",0)),int(meta.get("right",0))
            if (top+bottom+left+right)>0:
                H,W=bhwc.shape[1],bhwc.shape[2]
                ys=max(0,min(H,top)); ye=max(ys,min(H,H-bottom))
                xs=max(0,min(W,left)); xe=max(xs,min(W,W-right))
                bhwc=bhwc[:,ys:ye,xs:xe,:]
            H0,W0=int(meta.get("orig_h",bhwc.shape[1])),int(meta.get("orig_w",bhwc.shape[2]))
            bhwc=bhwc[:,:max(1,H0),:max(1,W0),:]
        if preserve_color:
            stats=latent.get("qi_color",None)
            if stats and "mean" in stats and "std" in stats:
                ref_mean=torch.tensor(stats["mean"],dtype=bhwc.dtype,device=bhwc.device)
                ref_std=torch.tensor(stats["std"],dtype=bhwc.dtype,device=bhwc.device)
                bhwc=_match_color(bhwc, ref_mean, ref_std)
        if move_to_cpu:
            bhwc=bhwc.cpu().contiguous()
        return (bhwc,)

__all__ = [
    "QI_TextEncodeQwenImageEdit_Safe",
    "QI_VAEDecodeHQ",
]
