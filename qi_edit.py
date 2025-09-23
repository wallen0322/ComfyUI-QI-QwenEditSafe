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
        # Default system body (no tags). Clear this field to disable custom template.
        default_template = (
            "You are a Prompt optimizer for image generation. Evaluate the user's input: "
            "if it requires expansion (e.g., due to vagueness, multiple characters without clear distinction, or missing details), "
            "enhance it by adding precise feature descriptions, using unique labels per character, ensuring visual distinctions, "
            "and including quality tags like \"Ultra HD, 4K, cinematic lighting\". "
            "If the input is already detailed and clear, output it verbatim. Keep enhanced prompts under 300 words."
        )

        return {"required":{
                    "clip":("CLIP",),
                    "prompt":("STRING",{"multiline":True,"default":""}),
                    "vae":("VAE",),
                    "image":("IMAGE",)},
                "optional":{
                    "image2": ("IMAGE",),
                    "image3": ("IMAGE",),
                    "image4": ("IMAGE",),
                    "image5": ("IMAGE",),
                    "no_resize_pad":("BOOLEAN",{"default":True}),
                    "pad_mode":(["reflect","replicate"],{"default":"reflect"}),
                    "grid_multiple":("INT",{"default":64,"min":8,"max":128,"step":8}),
                    "inject_mode":(["both","latents","pixels"],{"default":"both"}),
                    "encode_fp32":("BOOLEAN",{"default":True}),
                    "vl_max_pixels":("INT",{"default":16_777_216,"min":0,"max":16_777_216,"step":65536}),
                    "system_template":("STRING", {"multiline": True, "default": default_template}),
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
            "pixL_range": pix_late,  "pixL_w": float(w_pix_l)}

    def encode(self, clip, prompt, vae, image, 
               no_resize_pad=True, pad_mode="reflect", grid_multiple=64,
               inject_mode="both", encode_fp32=True,
               vl_max_pixels=16_777_216, system_template: Optional[str]=None, image2=None, image3=None, image4=None, image5=None):

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

        # Temporarily apply custom Qwen2.5VL system template for image prompts if supported
        original_tokenizer = getattr(clip, 'tokenizer', None)
        restore_needed = False
        old_template = None
        if original_tokenizer is not None and system_template is not None and len(system_template.strip()) > 0:
            try:
                # Many Qwen tokenizers read from `llama_template_images`
                sys_body = system_template.strip()
                if hasattr(original_tokenizer, 'llama_template_images'):
                    old_template = getattr(original_tokenizer, 'llama_template_images', None)
                    # Wrap user-provided system content with fixed Qwen image template skeleton
                    final_template = (
                        "<|im_start|>system\n" + sys_body +
                        "\n<|im_end|>\n<|im_start|>user\n"
                        "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    setattr(original_tokenizer, 'llama_template_images', final_template)
                    restore_needed = True
                elif hasattr(original_tokenizer, 'llama_template'):
                    # Fallback: some tokenizers only use text template key
                    old_template = getattr(original_tokenizer, 'llama_template', None)
                    final_template = (
                        "<|im_start|>system\n" + sys_body +
                        "\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                    )
                    setattr(original_tokenizer, 'llama_template', final_template)
                    restore_needed = True
            except Exception:
                restore_needed = False

        tokens=clip.tokenize(prompt, images=vl_imgs)
        # Restore tokenizer template if we changed it
        if restore_needed and original_tokenizer is not None:
            try:
                if hasattr(original_tokenizer, 'llama_template_images'):
                    setattr(original_tokenizer, 'llama_template_images', old_template)
                elif hasattr(original_tokenizer, 'llama_template'):
                    setattr(original_tokenizer, 'llama_template', old_template)
            except Exception:
                pass
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
        pix_base = recon_bhwc if (need_recon  and  recon_bhwc is not None) else padded
        pixE = _lowpass_ref(pix_base, 64) if need_pixels else None
        pixM = pix_base if need_pixels else None
        pixL = _hf_ref(pix_base, alpha=0.4, blur_k=3) if need_pixels else None

        # Use an internal default emphasis now that the external control is removed
        sch = self._derive_schedule(0.60, prompt, H, W)

        # latent anchors
        if inject_mode in ("both","latents"):
            for _ in range(sch["lat_rep"]):
                cond=node_helpers.conditioning_set_values(
                    cond, {"reference_latents": ref_latents,
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

        latent={"samples":lat}
        ref_latents = [lat]
        _REF_MAX_PIX = 1_200_000
        for _im in (image2, image3, image4, image5):
            if _im is None: continue
            _bh = _bhwc(_im)[...,:3]
            cur_area = _bh.shape[1]*_bh.shape[2]
            if cur_area > _REF_MAX_PIX:
                scale = (_REF_MAX_PIX/float(cur_area))**0.5
                Hs = max(1,int(_bh.shape[1]*scale)); Ws = max(1,int(_bh.shape[2]*scale))
                _bh = _ensure_rgb3(_resize_bchw_smart(_bchw(_bh), Ws, Hs).movedim(1,-1))
            _l = vae.encode(_bh)
            if isinstance(_l, dict) and 'samples' in _l: _l = _l['samples']
            if isinstance(_l, torch.Tensor) and _l.dtype != torch.float32: _l = _l.float()
            ref_latents.append(_l)
        return (cond, src, latent)

# ----------------- VAE decode (unchanged API) -----------------

__all__ = ["QI_TextEncodeQwenImageEdit_Safe"]
