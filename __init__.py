from typing import Tuple, List
import math, random
import torch
import torch.nn.functional as F
import node_helpers
import comfy.utils

# ======================================================
# 通用工具
# ======================================================
def _ensure_bhwc01(img: torch.Tensor) -> torch.Tensor:
    assert img.ndim == 4, f"Expect BHWC 4D, got {tuple(img.shape)}"
    if img.dtype != torch.float32:
        img = img.float()
    return img.clamp(0.0, 1.0).contiguous()

def _bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-1, 1).contiguous()

def _bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(1, -1).contiguous()

def _snap(v: int, m: int, policy: str) -> int:
    if m <= 1: return v
    if policy == "nearest":
        down = (v // m) * m
        up = down + m
        return up if (v - down) > (up - v) else down
    return (v // m) * m

def _scale_to_total(h: int, w: int, total_px: int) -> Tuple[int,int]:
    s = math.sqrt(max(total_px,1) / max(h*w,1))
    H = max(1, round(h * s))
    W = max(1, round(w * s))
    return H, W

def _resize_bchw(x: torch.Tensor, Wt: int, Ht: int, method: str) -> torch.Tensor:
    return comfy.utils.common_upscale(x, Wt, Ht, method, "disabled")

# 轻量 KMeans（在 torch 上跑，迭代次数有限）
def _kmeans_colors(img_bhwc: torch.Tensor, K: int=5, iters: int=6) -> torch.Tensor:
    # img_bhwc: 1 x H x W x 3  in [0,1]
    B,H,W,C = img_bhwc.shape
    assert B==1 and C==3
    x = img_bhwc.view(1, H*W, 3).squeeze(0)  # (N,3)
    N = x.shape[0]
    if N == 0:
        return torch.ones((K,3), device=x.device, dtype=x.dtype)
    # 初始化：随机采样 K 个像素
    idx = torch.randperm(N, device=x.device)[:K]
    cent = x[idx].clone()   # (K,3)
    for _ in range(max(1,iters)):
        # 分配
        # (N,K) 距离^2
        dist2 = ((x[:,None,:] - cent[None,:,:])**2).sum(-1)
        assign = dist2.argmin(dim=1)  # (N,)
        # 重算中心
        new_cent = []
        for k in range(K):
            m = (assign == k)
            if m.any():
                new_cent.append(x[m].mean(0))
            else:
                # 如果该簇空了，重新随机一个
                ridx = torch.randint(0, N, (1,), device=x.device)
                new_cent.append(x[ridx].squeeze(0))
        cent = torch.stack(new_cent, dim=0)
    # 排序：按亮度从低到高（避免颜色块抖动）
    lum = (0.2126*cent[:,0] + 0.7152*cent[:,1] + 0.0722*cent[:,2])
    order = torch.argsort(lum)
    cent = cent[order]
    return cent  # (K,3)

def _make_palette_swatch(centroids: torch.Tensor, tile: int=32, cols: int=None) -> torch.Tensor:
    # centroids: (K,3) in [0,1]; 生成 1 x H x W x 3 色块图
    K = centroids.shape[0]
    if cols is None:
        cols = min(K, 8)
    rows = (K + cols - 1) // cols
    H = rows * tile
    W = cols * tile
    sw = torch.zeros((1, H, W, 3), device=centroids.device, dtype=centroids.dtype)
    for i in range(K):
        r = i // cols; c = i % cols
        color = centroids[i].view(1,1,1,3)
        sw[:, r*tile:(r+1)*tile, c*tile:(c+1)*tile, :] = color
    return sw

# ======================================================
# 英文 Safe 节点（保留以便回退）
# ======================================================
CATEGORY_SAFE = "advanced/conditioning (QI Safe)"
_RESAMPLERS = ["lanczos","bicubic","area","nearest"]

class QI_TextEncodeQwenImageEdit_Safe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {
                "no_scale": ("BOOLEAN", {"default": False}),
                "target_total_pixels": ("INT", {"default": 1024*1024, "min": 256*256, "max": 4096*4096, "step": 65536}),
                "grid_multiple": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "snap_policy": (["down","nearest"], {"default": "down"}),
                "keep_alpha": ("BOOLEAN", {"default": False}),
                "resampler_down": (_RESAMPLERS, {"default": "area"}),
                "resampler_up": (_RESAMPLERS, {"default": "lanczos"}),
                "inject_mode": (["auto","latents","pixels","both"], {"default": "auto"}),
                "dual_image_tokenize": ("BOOLEAN", {"default": False}),
                "aux_total_pixels": ("INT", {"default": 512*512, "min": 256*256, "max": 1024*1024, "step": 65536}),
                "aux_repeat": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "hi_freq_boost": ("BOOLEAN", {"default": False}),
                "hf_radius": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2}),
                "hf_amount": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
                "nunchaku_compat": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "LATENT")
    RETURN_NAMES = ("conditioning", "image", "latent")
    FUNCTION = "encode"
    CATEGORY = CATEGORY_SAFE

    def encode(self, clip, prompt: str, image, vae,
               no_scale=False, target_total_pixels=1024*1024,
               grid_multiple=8, snap_policy="down",
               keep_alpha=False, resampler_down="area", resampler_up="lanczos",
               inject_mode="auto", dual_image_tokenize=False,
               aux_total_pixels=512*512, aux_repeat=1,
               hi_freq_boost=False, hf_radius=5, hf_amount=0.12,
               nunchaku_compat=True):

        grid = 64 if nunchaku_compat else grid_multiple

        img = _ensure_bhwc01(image)
        if not keep_alpha and img.shape[-1] == 4:
            img = img[..., :3]

        bchw = _bhwc_to_bchw(img)
        B,C,H,W = bchw.shape

        if no_scale:
            Ht = _snap(H, grid, snap_policy)
            Wt = _snap(W, grid, snap_policy)
            if (Ht, Wt) != (H, W):
                bchw = bchw[:, :, :Ht, :Wt]
        else:
            H1, W1 = _scale_to_total(H, W, target_total_pixels)
            H2 = _snap(H1, grid, snap_policy)
            W2 = _snap(W1, grid, snap_policy)
            if (H2, W2) != (H, W):
                method = resampler_down if (H2*W2 < H*W) else resampler_up
                bchw = _resize_bchw(bchw, W2, H2, method)

        aligned_image = _bchw_to_bhwc(bchw)[:, :, :, :3]

        if hi_freq_boost:
            aligned_image = _unsharp_mask_bhwc(aligned_image, radius=hf_radius, amount=hf_amount)

        images_list: List[torch.Tensor] = [aligned_image]
        if dual_image_tokenize and not nunchaku_compat:
            bh = _bhwc_to_bchw(img)
            H3, W3 = _scale_to_total(H, W, aux_total_pixels)
            H3 = _snap(H3, grid, snap_policy); W3 = _snap(W3, grid, snap_policy)
            bh = _resize_bchw(bh, W3, H3, resampler_down)
            aux_img = _bchw_to_bhwc(bh)[:, :, :, :3]
            for _ in range(max(1, aux_repeat)):
                images_list.append(aux_img)

        tokens = clip.tokenize(prompt, images=images_list)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latent = vae.encode(aligned_image)
        mode = inject_mode
        if inject_mode == "auto":
            mode = "pixels" if nunchaku_compat else "latents"

        if mode == "latents":
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        elif mode == "pixels":
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_pixels": [aligned_image]}, append=True)
        else:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_pixels": [aligned_image]}, append=True)

        latent_output = {"samples": ref_latent}
        return (conditioning, aligned_image, latent_output)

# ======================================================
# 中文一致性增强节点（带“调色板锚定”）
# ======================================================
CATEGORY_CN = "高级/条件（中文·一致性增强） (QI)"
_RESAMPLERS = ["lanczos","bicubic","area","nearest"]

class QI_TextEncodeQwenImageEdit_CN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CLIP模型": ("CLIP",),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "图像": ("IMAGE",),
                "VAE": ("VAE",),
            },
            "optional": {
                "不缩放_保持原尺寸": ("BOOLEAN", {"default": False}),
                "目标总像素": ("INT", {"default": 1024*1024, "min": 256*256, "max": 4096*4096, "step": 65536}),
                "网格倍数": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "对齐策略": (["向下对齐","就近对齐"], {"default": "向下对齐"}),
                "保留Alpha": ("BOOLEAN", {"default": False}),
                "下采样插值": (_RESAMPLERS, {"default": "area"}),
                "上采样插值": (_RESAMPLERS, {"default": "lanczos"}),
                "注入模式": (["自动","仅latent","仅像素","两者都注入"], {"default": "自动"}),
                "辅助原图参与编码": ("BOOLEAN", {"default": False}),
                "辅助原图像素": ("INT", {"default": 512*512, "min": 256*256, "max": 1024*1024, "step": 65536}),
                "辅助重复次数": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "调色板锚定": ("BOOLEAN", {"default": True}),
                "调色板颜色数": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "调色板重复次数": ("INT", {"default": 2, "min": 1, "max": 6, "step": 1}),
                "高频轻加强": ("BOOLEAN", {"default": False}),
                "锐化半径": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2}),
                "锐化强度": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
                "兼容Nunchaku_FLUX": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("CONDITIONING","IMAGE","LATENT")
    RETURN_NAMES = ("条件","图像","潜空间")
    FUNCTION = "编码"
    CATEGORY = CATEGORY_CN

    def 编码(self, CLIP模型, 提示词: str, 图像, VAE,
           不缩放_保持原尺寸=False, 目标总像素=1024*1024, 网格倍数=8, 对齐策略="向下对齐",
           保留Alpha=False, 下采样插值="area", 上采样插值="lanczos",
           注入模式="自动", 辅助原图参与编码=False, 辅助原图像素=512*512, 辅助重复次数=1,
           调色板锚定=True, 调色板颜色数=5, 调色板重复次数=2,
           高频轻加强=False, 锐化半径=5, 锐化强度=0.12,
           兼容Nunchaku_FLUX=True):

        snap_policy = "down" if 对齐策略=="向下对齐" else "nearest"
        grid = 64 if 兼容Nunchaku_FLUX else 网格倍数

        img = _ensure_bhwc01(图像)
        if not 保留Alpha and img.shape[-1] == 4:
            img = img[..., :3]

        bchw = _bhwc_to_bchw(img)
        B,C,H,W = bchw.shape

        if 不缩放_保持原尺寸:
            Ht = _snap(H, grid, snap_policy)
            Wt = _snap(W, grid, snap_policy)
            if (Ht, Wt) != (H, W):
                bchw = bchw[:, :, :Ht, :Wt]
        else:
            H1, W1 = _scale_to_total(H, W, 目标总像素)
            H2 = _snap(H1, grid, snap_policy)
            W2 = _snap(W1, grid, snap_policy)
            if (H2, W2) != (H, W):
                method = 下采样插值 if (H2*W2 < H*W) else 上采样插值
                bchw = _resize_bchw(bchw, W2, H2, method)

        aligned = _bchw_to_bhwc(bchw)[:, :, :, :3]

        if 高频轻加强:
            # 复用上面的轻量锐化实现
            aligned = self._unsharp(aligned, 锐化半径, 锐化强度)

        # 组装多图 tokens：主图 + （可选）原图 + （可选）调色板锚定图
        images_list: List[torch.Tensor] = [aligned]
        if 辅助原图参与编码 and not 兼容Nunchaku_FLUX:
            bh = _bhwc_to_bchw(img)
            H3, W3 = _scale_to_total(H, W, 辅助原图像素)
            H3 = _snap(H3, grid, snap_policy); W3 = _snap(W3, grid, snap_policy)
            bh = _resize_bchw(bh, W3, H3, 下采样插值)
            aux_img = _bchw_to_bhwc(bh)[:, :, :, :3]
            for _ in range(max(1, 辅助重复次数)):
                images_list.append(aux_img)

        if 调色板锚定 and not 兼容Nunchaku_FLUX:
            # 从“主对齐图”提取 K 色，生成色块图，重复 N 次，作为强色彩先验
            small = comfy.utils.common_upscale(_bhwc_to_bchw(aligned), 64, 64, "area", "disabled")
            small_bhwc = _bchw_to_bhwc(small)
            cents = _kmeans_colors(small_bhwc, K=int(调色板颜色数), iters=6)
            swatch = _make_palette_swatch(cents, tile=32, cols=min(调色板颜色数,8))
            for _ in range(max(1, 调色板重复次数)):
                images_list.append(swatch)

        tokens = CLIP模型.tokenize(提示词, images=images_list)
        conditioning = CLIP模型.encode_from_tokens_scheduled(tokens)

        # 用与主图同源像素生成 reference
        ref_latent = VAE.encode(aligned)

        mode = 注入模式
        if 注入模式 == "自动":
            mode = "仅像素" if 兼容Nunchaku_FLUX else "仅latent"

        if mode == "仅latent":
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        elif mode == "仅像素":
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_pixels": [aligned]}, append=True)
        else:  # 两者都注入
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_pixels": [aligned]}, append=True)

        latent_output = {"samples": ref_latent}
        return (conditioning, aligned, latent_output)

    # 轻量锐化（与英文版一致）
    def _unsharp(self, img: torch.Tensor, radius: int=5, amount: float=0.12) -> torch.Tensor:
        if radius <= 0 or amount <= 0.0:
            return img
        k = max(1, int(radius))
        if k % 2 == 0: k += 1
        pad = k // 2
        x = _bhwc_to_bchw(img)
        ker = torch.ones((1,1,k), dtype=x.dtype, device=x.device) / k
        for _ in range(3):
            x = F.pad(x, (pad,pad,0,0), mode="reflect")
            x = F.conv2d(x, ker.unsqueeze(2).expand(x.size(1),1,k,1), groups=x.size(1))
            x = F.pad(x, (0,0,pad,pad), mode="reflect")
            x = F.conv2d(x, ker.unsqueeze(3).expand(x.size(1),1,1,k), groups=x.size(1))
        blur = _bchw_to_bhwc(x)
        sharp = torch.clamp(img + amount * (img - blur), 0.0, 1.0)
        return sharp

NODE_CLASS_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": QI_TextEncodeQwenImageEdit_Safe,
    "QI_TextEncodeQwenImageEdit_CN": QI_TextEncodeQwenImageEdit_CN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QI_TextEncodeQwenImageEdit_Safe": "QI • TextEncodeQwenImageEdit (Safe)",
    "QI_TextEncodeQwenImageEdit_CN": "QI • 文生图编辑（中文·一致性增强）",
}
