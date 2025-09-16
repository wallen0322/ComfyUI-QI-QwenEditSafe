from __future__ import annotations
from typing import List
import torch, torch.nn.functional as F
import node_helpers, comfy.utils

# -------- helpers --------
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

# ---------- Encoder (Safe) ----------
class QI_TextEncodeQwenImageEdit_Safe:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("CONDITIONING","IMAGE","LATENT")
    RETURN_NAMES = ("conditioning","image","latent")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"clip":("CLIP",),"prompt":("STRING",{"multiline":True,"default":""}),"image":("IMAGE",),"vae":("VAE",)},
                "optional":{
                    "no_resize_pad":("BOOLEAN",{"default":True}),
                    "pad_mode":(["reflect","replicate"],{"default":"reflect"}),
                    "grid_multiple":("INT",{"default":64,"min":8,"max":128,"step":8}),

                    "inject_mode":(["both","latents","pixels"],{"default":"both"}),
                    "pixels_source":(["recon","input"],{"default":"recon"}),
                    "pixels_shaping":(["colorfield_64","colorfield_32","full"],{"default":"colorfield_64"}),
                    "encode_fp32":("BOOLEAN",{"default":True}),
                    "latent_lowpass_radius":("INT",{"default":0,"min":0,"max":21,"step":1}),

                    "prompt_emphasis":("FLOAT",{"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
                }}

    def _make_colorfield(self, img_bhwc: torch.Tensor, kind:str)->torch.Tensor:
        if kind=="full": return img_bhwc
        size=64 if kind=="colorfield_64" else 32
        bh=_resize_bchw(_bchw(img_bhwc),size,size,"area")
        bh=_sep_box_blur_bchw(bh,3)
        return _ensure_rgb3(_resize_bchw(bh,img_bhwc.shape[2],img_bhwc.shape[1],"area").movedim(1,-1)).clamp(0,1)

    def encode(self, clip, prompt, image, vae,
               no_resize_pad=True, pad_mode="reflect", grid_multiple=64,
               inject_mode="both", pixels_source="recon", pixels_shaping="colorfield_64",
               encode_fp32=True, latent_lowpass_radius=0, prompt_emphasis=0.5):

        # geometry (no-resize, symmetric pad)
        src=_bhwc(image)[...,:3]; B,H,W,C=src.shape
        Ht,Wt=_ceil_to(H,grid_multiple),_ceil_to(W,grid_multiple)
        bchw=_bchw(src); top=left=bottom=right=0
        if no_resize_pad:
            ph,pw=Ht-H,Wt-W
            if ph>0 or pw>0:
                top=ph//2; bottom=ph-top; left=pw//2; right=pw-left
                bchw=F.pad(bchw,(left,right,top,bottom),mode=pad_mode)
        padded=_ensure_rgb3(bchw.movedim(1,-1)).clamp(0,1)

        # clip tokens
        tokens=clip.tokenize(prompt, images=[padded])
        cond=clip.encode_from_tokens_scheduled(tokens)

        # vae encode/decode
        lat=vae.encode(padded)
        if encode_fp32 and isinstance(lat,torch.Tensor) and lat.dtype!=torch.float32:
            lat=lat.float()
        if isinstance(lat,torch.Tensor) and int(latent_lowpass_radius)>0:
            lat=_sep_box_blur_bchw(lat if (lat.ndim==4 and lat.shape[1] in (1,2,3,4,8,16,32)) else lat.movedim(-1,1),
                                   int(latent_lowpass_radius))
        with torch.no_grad():
            recon=vae.decode(lat)
        recon=_bhwc(recon)

        pix_base=recon if pixels_source=="recon" else padded
        pshape = "colorfield_32" if (pixels_shaping!="full" and prompt_emphasis>=0.6) else pixels_shaping
        pix_ref=self._make_colorfield(pix_base, pshape)

        lat_rep=1 + int(prompt_emphasis>=0.7)
        pix_rep=0 if prompt_emphasis>=0.8 else (1 if prompt_emphasis>=0.6 else 2)

        if inject_mode in ("both","latents"):
            for _ in range(lat_rep):
                cond=node_helpers.conditioning_set_values(cond,{"reference_latents":[lat]},append=True)
        if inject_mode in ("both","pixels"):
            for _ in range(pix_rep):
                cond=node_helpers.conditioning_set_values(cond,{"reference_pixels":[pix_ref]},append=True)

        latent={"samples":lat,"qi_pad":{"top":int(top),"bottom":int(bottom),"left":int(left),"right":int(right),
                                        "orig_h":int(H),"orig_w":int(W)}}
        return (cond, src, latent)

# ---------- Chinese mirror ----------
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
                    "像素来源":(["VAE重建","原始图像"],{"default":"VAE重建"}),
                    "像素形态":(["色彩场_64","色彩场_32","完整像素"],{"default":"色彩场_64"}),
                    "编码用FP32":("BOOLEAN",{"default":True}),
                    "潜空间低通半径":("INT",{"default":0,"min":0,"max":21,"step":1}),

                    "文本优先度":("FLOAT",{"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
                }}

    def 编码(self, CLIP模型, 提示词, 图像, VAE,
           模式_不缩放网格填充=True, 填充方式="reflect", 网格倍数=64,
           注入模式="两者都注入", 像素来源="VAE重建", 像素形态="色彩场_64",
           编码用FP32=True, 潜空间低通半径=0, 文本优先度=0.5):

        core=QI_TextEncodeQwenImageEdit_Safe()
        imap={"两者都注入":"both","仅latent":"latents","仅像素":"pixels"}
        psrc={"VAE重建":"recon","原始图像":"input"}
        pshape={"色彩场_64":"colorfield_64","色彩场_32":"colorfield_32","完整像素":"full"}
        return core.encode(CLIP模型, 提示词, 图像, VAE,
                           no_resize_pad=模式_不缩放网格填充, pad_mode=填充方式, grid_multiple=网格倍数,
                           inject_mode=imap.get(注入模式,"both"),
                           pixels_source=psrc.get(像素来源,"recon"),
                           pixels_shaping=pshape.get(像素形态,"colorfield_64"),
                           encode_fp32=编码用FP32, latent_lowpass_radius=潜空间低通半径,
                           prompt_emphasis=文本优先度)

# ---------- Decode ----------
class QI_VAEDecodeHQ:
    CATEGORY="QI by wallen0322"
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("image",); FUNCTION="decode"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"vae":("VAE",),"latent":("LATENT",)},"optional":{"force_fp32":("BOOLEAN",{"default":True})}}
    def decode(self, vae, latent, force_fp32=True):
        x=latent["samples"]
        if force_fp32 and isinstance(x,torch.Tensor) and x.dtype!=torch.float32: x=x.float()
        with torch.no_grad(): img=vae.decode(x)
        if isinstance(img,(list,tuple)): img=img[0]
        bhwc=_bhwc(img)
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
        return (bhwc.contiguous(),)

NODE_CLASS_MAPPINGS={
    "QI_TextEncodeQwenImageEdit_Safe":QI_TextEncodeQwenImageEdit_Safe,
    "QI_TextEncodeQwenImageEdit_CN":QI_TextEncodeQwenImageEdit_CN,
    "QI_VAEDecodeHQ":QI_VAEDecodeHQ,
}
NODE_DISPLAY_NAME_MAPPINGS={
    "QI_TextEncodeQwenImageEdit_Safe":"QI • TextEncodeQwenImageEdit — by wallen0322",
    "QI_TextEncodeQwenImageEdit_CN":"QI • 文生图编辑 — by wallen0322",
    "QI_VAEDecodeHQ":"QI • VAE Decode — by wallen0322",
}
