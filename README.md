# ComfyUI-QI-QwenEditSafe
ComfyUI-QI-QwenEditSafe

QI • 文生图编辑（中文·一致性增强）

QI • TextEncodeQwenImageEdit (Chinese · Consistency-Enhanced)

用途 | Purpose
在 Qwen Image Edit 编辑场景下，减少构图/姿态/纹理漂移，抑制冷暖色偏与“画质被洗”。通过推理阶段优化（而非后期滤镜）：让条件像素与 VAE 参考同源；支持 reference_latents / reference_pixels 注入；可加入辅助原图与调色板锚定图作为多图视觉 token，直接把风格与色域锚定在编码阶段。

关键点 | Key Ideas

同源像素：用于条件与 VAE 的图像一致（减少构图/缩放误差）。

参考注入：reference_latents / reference_pixels（可单独或同时注入）。

多图 token：可把轻下采样原图与调色板锚定图一并送入 tokenize(images=[...])，锁定身份、光影与色域。

分支兼容：提供双截棍/FLUX兼容模式（网格 64，优先像素注入），降低形状/视图错误。

参数对照（中英）| Parameters (CN ⇄ EN)
中文名	英文名	类型 / 默认	说明（中文）	Description (EN)
CLIP模型	clip	CLIP	输入 CLIP 对象	Input CLIP object
提示词	prompt	string / ""	文本提示，可中英	Text prompt
图像	image	IMAGE	输入图（BHWC，[0,1]）	Input image
VAE	vae	VAE	输入 VAE 对象	VAE object
不缩放_保持原尺寸	no_scale	bool / False	不缩放，仅网格对齐裁切，保真度最高	No resize; grid-aligned crop
目标总像素	target_total_pixels	int / 1_048_576	需缩放时的目标像素（≈1MP）	Target pixels when resizing
网格倍数	grid_multiple	int / 8	分辨率对齐倍数；双截棍/FLUX 建议 64	Snap to multiple; 64 for Nunchaku/FLUX
对齐策略	snap_policy	enum / `向下对齐	就近对齐`	就近更锐，向下更稳
保留Alpha	keep_alpha	bool / False	保留 alpha 或强制 RGB3	Keep alpha / force RGB3
下采样插值	resampler_down	enum / area	下采样插值；area 稳定	Downscale resampler
上采样插值	resampler_up	enum / lanczos	上采样插值；lanczos 清晰	Upscale resampler
注入模式	inject_mode	enum / `自动	仅latent	仅像素
辅助原图参与编码	dual_image_tokenize	bool / False	轻下采样原图也作为视觉 token（非双截棍/FLUX 场景推荐开）	Feed downsampled original as extra token
辅助原图像素	aux_total_pixels	int / 512×512	辅助原图目标像素	Target pixels of the auxiliary image
辅助重复次数	aux_repeat	int / 1	重复等于“加权”	Repeat = weighting
调色板锚定	palette anchoring	bool / True	从原图聚类 K 色块生成“色板”token，锁色域	KMeans palette swatch as visual prior
调色板颜色数	palette_k	int / 5	主色数量	Number of dominant colors
调色板重复次数	palette_repeat	int / 2	色板重复次数	Palette repeats
高频轻加强	hi_freq_boost	bool / False	编码前极轻锐化（非后期）	Mild pre-encode unsharp
锐化半径	hf_radius	int / 5	高频加强半径	Unsharp radius
锐化强度	hf_amount	float / 0.12	高频加强强度	Unsharp amount
兼容双截棍/FLUX	nunchaku_compat	bool / True	兼容模式：网格 64，默认像素注入	Compatibility: grid 64; pixels preferred

注 / Note：注入模式=自动 时，标准 Qwen 默认注入 latents；双截棍/FLUX 默认注入 pixels。

推荐预设 | Recommended Presets

标准 Qwen（锁色 + 保真） / Standard Qwen (color-locked + fidelity)

不缩放_保持原尺寸：True；网格倍数：8；对齐：就近对齐

注入模式：两者都注入

辅助原图参与编码：True；辅助像素：512×512；辅助重复：2

调色板锚定：True；颜色数：5；重复：2

采样建议：steps=22–28, cfg=3.6–4.2, denoise=0.25–0.35

双截棍/FLUX（稳定优先） / Nunchaku/FLUX (stability-first)

兼容双截棍/FLUX：True（网格 64）

不缩放_保持原尺寸：True（或 1.2MP 目标总像素）；对齐：向下对齐

注入模式：仅像素（或 两者都注入 试验）

辅助原图参与编码：False；调色板锚定：False（稳定后可尝试开启）

采样建议：steps=22–26, cfg=3.6–4.2, denoise≈0.28

故障排查 | Troubleshooting

shape/view 报错 → 开启兼容双截棍/FLUX（网格 64），或手动将网格倍数设为 64。

冷色变暖色 → 开启调色板锚定（颜色数 5、重复 2），用 两者都注入（标准 Qwen）或 仅像素（双截棍/FLUX），并适当降低 denoise。

细节发糊 → 用 就近对齐、开启高频轻加强（半径 5、强度 0.10–0.15），并增大“目标总像素”。
