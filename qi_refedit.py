from __future__ import annotations
import torch
import torch.nn.functional as F
import node_helpers, comfy.utils

def _to_bhwc_any(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, (list, tuple)): x = x[0]
    if x.ndim == 3:
        if x.shape[0] in (1,3,4): x = x.unsqueeze(0).movedim(1,-1)
        elif x.shape[-1] in (1,3,4): x = x.unsqueeze(0)
        else: x = x.unsqueeze(0).movedim(1,-1)
    elif x.ndim == 4:
        if x.shape[-1] not in (1,3,4) and x.shape[1] in (1,3,4): x = x.movedim(1,-1)
    elif x.ndim == 5:
        if x.shape[2] in (1,3,4):
            B,T,C,H,W = x.shape
            x = x.reshape(B*T, C, H, W).movedim(1,-1)
        elif x.shape[-1] in (1,3,4):
            B,H,W,C,T = x.shape
            x = x.permute(0,4,1,2,3).reshape(B*T, H, W, C)
        else:
            B,C,H,W,T = x.shape
            x = x.permute(0,4,1,2,3).reshape(B*T, C, H, W).movedim(1,-1)
    else:
        raise RuntimeError(f"Unsupported ndim {x.ndim}")
    return x.float().clamp(0,1).contiguous()

def _ensure_rgb3(bhwc: torch.Tensor) -> torch.Tensor:
    c = int(bhwc.shape[-1])
    if c == 3: return bhwc
    if c == 1: return bhwc.repeat(1,1,1,3)
    return bhwc[...,:3]

def _bhwc(x: torch.Tensor) -> torch.Tensor: return _ensure_rgb3(_to_bhwc_any(x))
def _bchw(x: torch.Tensor) -> torch.Tensor: return x.movedim(-1,1).contiguous()
def _ceil_to(v:int, m:int)->int: return ((v+m-1)//m)*m if m>1 else v
def _choose_method(sw:int, sh:int, tw:int, th:int)->str:
    if tw*th<=0: return "area"
    if tw>=sw or th>=sh: return "lanczos"
    return "area"
def _resize_bchw(x:torch.Tensor,w:int,h:int,method:str)->torch.Tensor:
    return comfy.utils.common_upscale(x,w,h,method,"disabled")
def _resize_bchw_smart(x:torch.Tensor,w:int,h:int)->torch.Tensor:
    _,_,H,W = x.shape
    return _resize_bchw(x,w,h,_choose_method(W,H,w,h))

def _box_blur3(bchw: torch.Tensor, k:int=3)->torch.Tensor:
    if k<=1: return bchw
    if k%2==0: k+=1
    pad=k//2; B,C,H,W=bchw.shape
    if H<=pad or W<=pad: return bchw
    ker=torch.ones((C,1,k,k),dtype=bchw.dtype,device=bchw.device)/(k*k)
    return F.conv2d(F.pad(bchw,(pad,pad,pad,pad),mode="reflect"), ker, groups=C)

def _lowpass_ref(bhwc: torch.Tensor, size:int=64)->torch.Tensor:
    bchw=_bchw(bhwc)
    bh=_resize_bchw(bchw, size, size, "area")
    bh=_resize_bchw_smart(bh, bchw.shape[-1], bchw.shape[-2])
    return _ensure_rgb3(bh.movedim(1,-1)).clamp(0,1)

def _rgb_to_luma(bhwc: torch.Tensor)->torch.Tensor:
    bchw=_bchw(bhwc); r,g,b=bchw[:,0:1],bchw[:,1:2],bchw[:,2:3]
    return 0.299*r+0.587*g+0.114*b

def _sobel_luma(bhwc: torch.Tensor)->torch.Tensor:
    y=_rgb_to_luma(bhwc)
    ky=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=y.dtype,device=y.device).view(1,1,3,3)/8.0
    kx=ky.transpose(-1,-2)
    gx=F.conv2d(F.pad(y,(1,1,1,1),mode="replicate"), kx)
    gy=F.conv2d(F.pad(y,(1,1,1,1),mode="replicate"), ky)
    mag=torch.sqrt(gx*gx+gy*gy)+1e-6
    e=(mag/(mag.mean(dim=(2,3),keepdim=True)*3.0+1e-6)).clamp(0,1)
    return e

def _hf_ref_smart(bhwc: torch.Tensor, alpha: float, blur_k:int, kstd: float, bias: float, smooth_k:int)->torch.Tensor:
    bchw=_bchw(bhwc)
    base=_box_blur3(bchw, blur_k)
    detail=bchw-base
    k=3
    ker=torch.ones((1,1,k,k),dtype=bchw.dtype,device=bchw.device)/(k*k)
    y=_rgb_to_luma(bhwc)
    mu=F.conv2d(F.pad(y,(k//2,)*4,mode="replicate"), ker)
    var=F.conv2d(F.pad((y-mu)**2,(k//2,)*4,mode="replicate"), ker)
    std=torch.sqrt(var+1e-6)
    t=kstd*std+bias
    absd=torch.abs(detail)
    dshrink=torch.sign(detail)*torch.clamp(absd - torch.cat([t,t,t],dim=1), min=0.0)
    da=torch.abs(dshrink)
    cap=da.mean(dim=(2,3),keepdim=True)+2.2*da.std(dim=(2,3),keepdim=True)
    dshrink=torch.clamp(dshrink,min=-cap,max=cap)
    if smooth_k>1:
        if smooth_k%2==0: smooth_k+=1
        pad=smooth_k//2
        ker_s=torch.ones((3,1,smooth_k,smooth_k),dtype=bchw.dtype,device=bchw.device)/(smooth_k*smooth_k)
        dshrink=F.conv2d(F.pad(dshrink,(pad,pad,pad,pad),mode="replicate"), ker_s, groups=3)
    e=_sobel_luma(bhwc)
    tex=(std/(std.mean(dim=(2,3),keepdim=True)+1e-6)).clamp(0,1.5)
    tex=(tex-0.2)/0.8; tex=tex.clamp(0,1)
    gate=0.26+0.60*e+0.14*tex
    gate3=torch.cat([gate,gate,gate],dim=1)
    detail_mod = torch.tanh(dshrink * 2.0) * 0.5
    out=(bchw+alpha*(detail_mod*gate3)).clamp(0,1)
    return _ensure_rgb3(out.movedim(1,-1))

def _rng(a: float,b: float):
    a=float(max(0.0,min(1.0,a))); b=float(max(0.0,min(1.0,b))); return (a,b) if b>=a else (b,a)

class QI_RefEditEncode_Safe:
    CATEGORY = "QI by wallen0322"
    RETURN_TYPES = ("CONDITIONING","IMAGE","LATENT")
    RETURN_NAMES = ("conditioning","image","latent")
    FUNCTION = "encode"
    _ALIGN_MULTIPLE = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "clip":("CLIP",),
            "prompt":("STRING",{"multiline":True,"default":""}),
            "image":("IMAGE",),
            "vae":("VAE",),
            "prompt_emphasis": ("FLOAT", {"default":0.6, "min":0.0, "max":1.0, "step":0.01}),
            "quality_mode":(["natural","fast","balanced","best"],{"default":"natural"}),
        }}

    def encode(self, clip, prompt, image, vae, prompt_emphasis=0.6, quality_mode="natural"):
        early_rng = _rng(0.0, 0.6)
        late_rng  = _rng(0.6, 1.0)
        hf_rng    = _rng(0.985, 1.0)

        pad_mode="reflect"
        grid_multiple=32
        inject_mode="both"
        vl_max_pixels=16_777_216
        encode_fp32=False
        no_resize_pad=True

        emph=float(max(0.0,min(1.0,prompt_emphasis)))
        e_scale = 0.85 + 0.35*emph
        l_scale = 1.05 - 0.20*emph

        if quality_mode=="fast":
            lat_e, pixM_e, pixE_e = 0.70, 0.45, 0.22
            lat_l, pixM_l, pixL_l = 0.95, 0.78, 0.03
            hf_alpha, hf_kstd, hf_bias, hf_smooth = 0.16, 0.85, 0.0045, 5
        elif quality_mode=="best":
            lat_e, pixM_e, pixE_e = 0.78, 0.55, 0.28
            lat_l, pixM_l, pixL_l = 1.06, 0.90, 0.05
            hf_alpha, hf_kstd, hf_bias, hf_smooth = 0.20, 0.90, 0.0045, 7
        elif quality_mode=="balanced":
            lat_e, pixM_e, pixE_e = 0.74, 0.50, 0.25
            lat_l, pixM_l, pixL_l = 1.00, 0.85, 0.04
            hf_alpha, hf_kstd, hf_bias, hf_smooth = 0.18, 0.85, 0.0045, 5
        else:
            lat_e, pixM_e, pixE_e = 0.75, 0.50, 0.25
            lat_l, pixM_l, pixL_l = 1.02, 0.86, 0.04
            hf_alpha, hf_kstd, hf_bias, hf_smooth = 0.18, 0.85, 0.0045, 5

        lat_e *= e_scale; pixM_e *= e_scale
        lat_l *= l_scale; pixM_l *= l_scale

        src=_bhwc(image)[...,:3]
        H,W=int(src.shape[1]),int(src.shape[2])
        gm=max(8,int(grid_multiple))
        if gm%self._ALIGN_MULTIPLE!=0:
            gm=_ceil_to(gm,self._ALIGN_MULTIPLE)
        Ht=_ceil_to(H,gm); Wt=_ceil_to(W,gm)

        bchw=_bchw(src); top=left=bottom=right=0
        if no_resize_pad:
            ph,pw=Ht-H,Wt-W
            if ph>0 or pw>0:
                top=ph//2; bottom=ph-top; left=pw//2; right=pw-left
                mode = "replicate" if (bchw.shape[2]<=1 or bchw.shape[3]<=1) else pad_mode
                bchw=F.pad(bchw,(left,right,top,bottom),mode=mode)
        padded=_ensure_rgb3(bchw.movedim(1,-1)).clamp(0,1)

        vl_img=padded
        if vl_max_pixels and (padded.shape[1]*padded.shape[2] > vl_max_pixels):
            scale=(vl_max_pixels/float(padded.shape[1]*padded.shape[2]))**0.5
            Hs=max(1,int(padded.shape[1]*scale)); Ws=max(1,int(padded.shape[2]*scale))
            Hs=_ceil_to(Hs,8); Ws=_ceil_to(Ws,8)
            vl_img=_ensure_rgb3(_resize_bchw_smart(_bchw(padded),Ws,Hs).movedim(1,-1))
        tokens=clip.tokenize(prompt, images=[vl_img])
        cond=clip.encode_from_tokens_scheduled(tokens)

        with torch.inference_mode():
            lat=vae.encode(padded)
        if encode_fp32 and isinstance(lat,torch.Tensor) and lat.dtype!=torch.float32:
            lat=lat.float()

        need_pixels = inject_mode in ("both","pixels")
        pix_base = padded if need_pixels else None
        pixE = _lowpass_ref(pix_base,64) if (need_pixels and pixE_e>0) else None
        pixM = pix_base if (need_pixels and (pixM_e>0 or pixM_l>0)) else None
        pixL = _hf_ref_smart(pix_base, hf_alpha, 3, hf_kstd, hf_bias, hf_smooth) if (need_pixels and pixL_l>0) else None

        def add_ref(c, rng, lat_w, pixE_w, pixM_w, pixL_w, hf_rng=None):
            c=node_helpers.conditioning_set_values(c,{
                "reference_latents":[lat],
                "strength": float(lat_w),
                "timestep_percent_range":[float(rng[0]), float(rng[1])],
            }, append=True)
            if pixE is not None and pixE_w>0:
                c=node_helpers.conditioning_set_values(c,{
                    "reference_pixels":[pixE],
                    "strength": float(pixE_w),
                    "timestep_percent_range":[float(rng[0]), float(rng[1])],
                }, append=True)
            if pixM is not None and pixM_w>0:
                c=node_helpers.conditioning_set_values(c,{
                    "reference_pixels":[pixM],
                    "strength": float(pixM_w),
                    "timestep_percent_range":[float(rng[0]), float(rng[1])],
                }, append=True)
            if pixL is not None and pixL_w>0:
                hr = hf_rng if hf_rng is not None else rng
                c=node_helpers.conditioning_set_values(c,{
                    "reference_pixels":[pixL],
                    "strength": float(pixL_w),
                    "timestep_percent_range":[float(hr[0]), float(hr[1])],
                }, append=True)
            return c

        with torch.inference_mode():
            cond=add_ref(cond, early_rng, lat_e, pixE_e, pixM_e, 0.0)
            cond=add_ref(cond, late_rng,  lat_l, 0.0,    pixM_l, pixL_l, hf_rng)

        latent={
            "samples": lat,
            "qi_pad": {
                "top":int(top),"bottom":int(bottom),"left":int(left),"right":int(right),
                "orig_h":int(H),"orig_w":int(W)
            }
        }
        return (cond, _ensure_rgb3(_bhwc(image)), latent)
