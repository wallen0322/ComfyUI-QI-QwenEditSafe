from __future__ import annotations
from typing import Optional, Tuple
import torch, torch.nn.functional as F
import node_helpers, comfy.utils

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

def _resize_bchw(x:torch.Tensor,Wt:int,Ht:int,method:str)->torch.Tensor:
    return comfy.utils.common_upscale(x,Wt,Ht,method,"disabled")

def _sep_box_blur_bchw(x:torch.Tensor,k:int)->torch.Tensor:
    if k<=1: return x
    if k%2==0: k+=1
    pad=k//2; C=x.shape[1]
    ker=torch.ones((C,1,k,1),dtype=x.dtype,device=x.device)/k
    x=F.pad(x,(0,0,pad,pad),mode="reflect"); x=F.conv2d(x,ker,groups=C)
    ker=torch.ones((C,1,1,k),dtype=x.dtype,device=x.device)/k
    x=F.pad(x,(pad,pad,0,0),mode="reflect"); x=F.conv2d(x,ker,groups=C)
    return x

def _limit_area_keep_ar(bhwc: torch.Tensor, max_pixels: int) -> torch.Tensor:
    if max_pixels is None or max_pixels <= 0: return bhwc
    B,H,W,C = bhwc.shape
    area = H*W
    if area <= max_pixels: return bhwc
    scale = (max_pixels/float(area))**0.5
    Ht, Wt = max(1,int(H*scale)), max(1,int(W*scale))
    bchw = _bchw(bhwc)
    bchw = _resize_bchw(bchw, Wt, Ht, "area")
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

def _clampf(x: float, lo: float, hi: float) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

class QI_TextEncodeQwenImageEdit_Safe:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("CONDITIONING","IMAGE","LATENT")
    RETURN_NAMES = ("conditioning","image","latent")
    FUNCTION = "encode"

    _DEF_PIXELS_SOURCE = "recon"
    _DEF_PIXELS_SHAPING = "full"
    _DEF_LATENT_LOWPASS = 0
    _DEF_CONSISTENCY   = "sde_preserve"
    _DEF_EDIT_STRENGTH = 0.50
    _DEF_EDGE_KEEP     = 0.50
    _DEF_LATENT_W      = 1.00
    _DEF_PIXEL_W       = 0.50

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

    def _derive_schedule(self, emph: float):
        base_lat_w = self._DEF_LATENT_W
        base_pix_w = self._DEF_PIXEL_W
        base_edit  = self._DEF_EDIT_STRENGTH
        lat_w = _clampf(base_lat_w * (1.0 + 0.8*(emph-0.5)), 0.8, 1.6)
        pix_w = _clampf(base_pix_w * (1.0 - 0.6*(emph-0.5)), 0.2, 1.0)
        lat_rep = 1 + (1 if emph>=0.6 else 0) + (1 if emph>=0.8 else 0)
        pix_rep = max(0, 2 - (1 if emph>=0.6 else 0) - (1 if emph>=0.8 else 0))
        edit = _clampf(base_edit + 0.6*(emph-0.5), 0.0, 1.0)
        lat_end   = _clampf(0.25 + 0.45*edit, 0.10, 0.85)
        pix_start = _clampf(0.45 + 0.35*(1.0-edit), 0.30, 0.95)
        return {"lat_w":float(lat_w), "pix_w":float(pix_w),
                "lat_rep":int(lat_rep), "pix_rep":int(pix_rep),
                "lat_range":[0.0,float(lat_end)], "pix_range":[float(pix_start),1.0]}

    def _make_colorfield(self, img_bhwc: torch.Tensor, kind:str)->torch.Tensor:
        if kind=="full": return img_bhwc
        size=64 if kind=="colorfield_64" else 32
        bh=_resize_bchw(_bchw(img_bhwc),size,size,"area")
        bh=_sep_box_blur_bchw(bh,3)
        return _ensure_rgb3(_resize_bchw(bh,img_bhwc.shape[2],img_bhwc.shape[1],"area").movedim(1,-1)).clamp(0,1)

    def encode(self, clip, prompt, image, vae,
               no_resize_pad=True, pad_mode="reflect", grid_multiple=64,
               inject_mode="both", encode_fp32=True, prompt_emphasis=0.60,
               vl_max_pixels=16_777_216):

        src=_bhwc(image)[...,:3]
        src_mean, src_std = _rgb_stats(src)
        H,W=src.shape[1],src.shape[2]
        Ht,Wt=_ceil_to(H,grid_multiple),_ceil_to(W,grid_multiple)
        bchw=_bchw(src); top=left=bottom=right=0
        if no_resize_pad:
            ph,pw=Ht-H,Wt-W
            if ph>0 or pw>0:
                top=ph//2; bottom=ph-top; left=pw//2; right=pw-left
                bchw=F.pad(bchw,(left,right,top,bottom),mode=pad_mode)
        padded=_ensure_rgb3(bchw.movedim(1,-1)).clamp(0,1)

        vl_img = _limit_area_keep_ar(padded, int(vl_max_pixels) if vl_max_pixels is not None else 0)
        tokens=clip.tokenize(prompt, images=[vl_img])
        cond=clip.encode_from_tokens_scheduled(tokens)

        lat=vae.encode(padded)
        if encode_fp32 and isinstance(lat,torch.Tensor) and lat.dtype!=torch.float32:
            lat=lat.float()
        if isinstance(lat,torch.Tensor) and int(self._DEF_LATENT_LOWPASS)>0:
            lat=_sep_box_blur_bchw(lat if (lat.ndim==4 and lat.shape[1] in (1,2,3,4,8,16,32)) else lat.movedim(-1,1),
                                   int(self._DEF_LATENT_LOWPASS))

        need_pixels = inject_mode in ("both","pixels")
        need_recon  = need_pixels and (self._DEF_PIXELS_SOURCE=="recon")
        recon_bhwc: Optional[torch.Tensor] = None
        if need_recon:
            with torch.inference_mode():
                recon = vae.decode(lat)
            recon_bhwc = _bhwc(recon)
        pix_base = recon_bhwc if (need_recon and recon_bhwc is not None) else padded
        pix_ref = self._make_colorfield(pix_base, self._DEF_PIXELS_SHAPING) if need_pixels else None

        sch = self._derive_schedule(float(prompt_emphasis))
        if inject_mode in ("both","latents"):
            for _ in range(sch["lat_rep"]):
                cond=node_helpers.conditioning_set_values(
                    cond, {"reference_latents":[lat],
                           "strength": sch["lat_w"],
                           "timestep_percent_range": sch["lat_range"]},
                    append=True)
        if need_pixels and pix_ref is not None:
            for _ in range(sch["pix_rep"]):
                cond=node_helpers.conditioning_set_values(
                    cond, {"reference_pixels":[pix_ref],
                           "strength": sch["pix_w"],
                           "timestep_percent_range": sch["pix_range"]},
                    append=True)

        latent={"samples":lat,
                "qi_pad":{"top":int(top),"bottom":int(bottom),"left":int(left),"right":int(right),
                          "orig_h":int(H),"orig_w":int(W)},
                "qi_color":{"mean":src_mean.cpu().numpy().tolist(), "std":src_std.cpu().numpy().tolist()}}
        return (cond, src, latent)

class QI_TextEncodeQwenImageEdit_CN:
    CATEGORY="QI by wallen0322"
    RETURN_TYPES=("CONDITIONING","IMAGE","LATENT")
    RETURN_NAMES=("条件","图像","潜空间")
    FUNCTION="编码"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"CLIP模型":("CLIP",),"提示词":("STRING",{"multiline":True,"default":""}),"图像":("IMAGE",),"VAE":("VAE",)},
                "optional":{
                    "模式_不缩放网格填充":("BOOLEAN",{"default":True}),
                    "填充方式":(["reflect","replicate"],{"default":"reflect"}),
                    "网格倍数":("INT",{"default":64,"min":8,"max":128,"step":8}),
                    "注入模式":(["两者都注入","仅latent","仅像素"],{"default":"两者都注入"}),
                    "编码用FP32":("BOOLEAN",{"default":True}),
                    "文本优先度":("FLOAT",{"default":0.60,"min":0.0,"max":1.0,"step":0.05}),
                    "VL像素上限":("INT",{"default":16_777_216,"min":0,"max":16_777_216,"step":65536}),
                }}

    def 编码(self, CLIP模型, 提示词, 图像, VAE,
           模式_不缩放网格填充=True, 填充方式="reflect", 网格倍数=64,
           注入模式="两者都注入", 编码用FP32=True, 文本优先度=0.60, VL像素上限=16_777_216):

        core=QI_TextEncodeQwenImageEdit_Safe()
        imap={"两者都注入":"both","仅latent":"latents","仅像素":"pixels"}
        return core.encode(CLIP模型, 提示词, 图像, VAE,
                           no_resize_pad=模式_不缩放网格填充, pad_mode=填充方式, grid_multiple=网格倍数,
                           inject_mode=imap.get(注入模式,"both"),
                           encode_fp32=编码用FP32, prompt_emphasis=文本优先度, vl_max_pixels=VL像素上限)

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

class QI_VAEDecodeHQ_CN:
    CATEGORY="QI by wallen0322"
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("图像",); FUNCTION="解码"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"VAE":("VAE",),"潜空间":("LATENT",)},
                "optional":{"使用FP32":("BOOLEAN",{"default":True}),
                            "保留原色调":("BOOLEAN",{"default":True}),
                            "锐化强度":("FLOAT",{"default":0.0,"min":0.0,"max":1.0,"step":0.05}),
                            "解码移至CPU":("BOOLEAN",{"default":True}),}}
    def 解码(self, VAE, 潜空间, 使用FP32=True, 保留原色调=True, 锐化强度=0.0, 解码移至CPU=True):
        core=QI_VAEDecodeHQ()
        return core.decode(VAE, 潜空间, force_fp32=使用FP32, preserve_color=保留原色调,
                           unsharp_amount=锐化强度, move_to_cpu=解码移至CPU)

__all__ = [
    "QI_TextEncodeQwenImageEdit_Safe",
    "QI_TextEncodeQwenImageEdit_CN",
    "QI_VAEDecodeHQ",
    "QI_VAEDecodeHQ_CN",
]
