use candle_core::{Result, Tensor, D, DType};
use candle_nn::{Conv2dConfig, VarBuilder};

/// VAE configuration for Qwen-Image-2.1 (`AutoencoderKLQwenImage21`, decoder side).
///
/// Values match the real checkpoint's `vae/config.json`. This is a causal-3D VAE
/// (Wan-style) that degenerates to a plain 2D decoder for single-frame images:
/// `QwenImage21CausalConv3d` explicitly folds away the temporal axis when there is
/// no frame history to cache, so every conv here is an ordinary `Conv2d`.
#[derive(Debug, Clone)]
pub struct Config {
    pub z_dim: usize,
    /// Encoder base width (`base_dim`); the decoder uses `decoder_base_dim`.
    pub encoder_base_dim: usize,
    /// Encoder input channels: RGBA, matching the decoder's `out_channels`.
    pub in_channels: usize,
    pub decoder_base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Per-transition temporal-downsample flags (encoder side); reversed for the decoder.
    pub temperal_downsample: Vec<bool>,
    pub out_channels: usize,
    pub scale_factor_spatial: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            z_dim: 64,
            encoder_base_dim: 96,
            in_channels: 4,
            decoder_base_dim: 144,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 2,
            temperal_downsample: vec![false, true, true, true],
            out_channels: 4,
            scale_factor_spatial: 16,
            latents_mean: vec![
                0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510, 0.2732,
                0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
                -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636,
                3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433,
                0.3764, 0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373,
                1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
            ],
            latents_std: vec![
                3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904,
                2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
                3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075,
                3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
                3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
                3.7662, 3.6746, 3.4563, 3.8161,
            ],
        }
    }
}

impl Config {
    pub fn spatial_compression_ratio(&self) -> usize {
        self.scale_factor_spatial
    }
}

/// RMSNorm over the channel axis of a channel-first `[B, C, H, W]` tensor:
/// `x / sqrt(mean(x^2, dim=C) + eps) * gamma`. Matches `QwenImage21RMS_norm`
/// (its `F.normalize(x) * sqrt(dim)` formulation is algebraically identical to this).
#[derive(Debug, Clone)]
struct RmsNormChannelFirst {
    gamma: Tensor,
    dim: usize,
}

impl RmsNormChannelFirst {
    /// `trailing_ones` matches the source's `images` flag: 3 (`[dim,1,1,1]`) for
    /// `images=False` (every ResidualBlock norm, `decoder.head.0`), 2 (`[dim,1,1]`)
    /// for `images=True` (the AttentionBlock norm) — both have exactly `dim` elements.
    fn new(dim: usize, trailing_ones: usize, vb: VarBuilder) -> Result<Self> {
        let mut shape = vec![dim];
        shape.extend(std::iter::repeat(1).take(trailing_ones));
        let gamma = vb.get(shape, "gamma")?.reshape(dim)?;
        Ok(Self { gamma, dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let ms = x32.sqr()?.mean(1)?; // [B, H, W]
        let rms = ms.affine(1.0, 1e-6)?.sqrt()?.unsqueeze(1)?; // [B, 1, H, W]
        let normed = x32.broadcast_div(&rms)?;
        let gamma = self.gamma.reshape((1, self.dim, 1, 1))?;
        normed.broadcast_mul(&gamma)?.to_dtype(dt)
    }
}

/// Load a 2D conv. Two checkpoint layouts exist for this VAE:
/// - **native** (ComfyUI-repackaged, e.g. the community `*_vae_bf16.safetensors`
///   files): every conv weight is a genuine 5D Conv3d tensor `[out,in,1,kh,kw]`
///   (even pointwise ones) — squeeze away the trivial temporal dim.
/// - **diffusers** (the official `Qwen/Qwen-Image-2.1` `vae/` checkpoint): plain
///   4D `[out,in,kh,kw]` Conv2d weights, loadable directly.
/// Both stem from the same `QwenImage21CausalConv3d`, which folds away the
/// temporal axis for single-frame images either way — this is just a storage
/// difference between the two distributions of the same weights.
fn load_conv(native: bool, in_c: usize, out_c: usize, k: usize, padding: usize, vb: VarBuilder) -> Result<candle_nn::Conv2d> {
    if native {
        let ws = vb.get((out_c, in_c, 1, k, k), "weight")?.squeeze(2)?;
        let bs = vb.get(out_c, "bias")?;
        Ok(candle_nn::Conv2d::new(ws, Some(bs), Conv2dConfig { padding, ..Default::default() }))
    } else {
        candle_nn::conv2d(in_c, out_c, k, Conv2dConfig { padding, ..Default::default() }, vb)
    }
}

/// Upper bound on one convolution's im2col buffer. candle's conv2d (without
/// cuDNN) materialises `C_in·k·k·H·W` values: at 1024² the decoder's last
/// 288→144 3x3 conv alone would take ~10.9 GB in F32, which made the VAE decode
/// the memory peak of the whole pipeline.
const CONV_IM2COL_BUDGET_BYTES: usize = 1 << 30;

/// `x.apply(conv)`, computed in horizontal bands when its im2col buffer would
/// exceed the budget. Exact: the input is zero-padded once and each band reads
/// `k - 1` halo rows. Only stride-1, undilated, ungrouped convs are banded.
fn conv2d_banded(x: &Tensor, conv: &candle_nn::Conv2d) -> Result<Tensor> {
    conv2d_banded_with_budget(x, conv, CONV_IM2COL_BUDGET_BYTES)
}

fn conv2d_banded_with_budget(x: &Tensor, conv: &candle_nn::Conv2d, budget_bytes: usize) -> Result<Tensor> {
    let cfg = conv.config();
    let (_, c_in, h, w) = x.dims4()?;
    let k = conv.weight().dim(2)?;
    let p = cfg.padding;
    let (h_out, w_out) = (h + 2 * p + 1 - k, w + 2 * p + 1 - k);
    let row_bytes = c_in * k * k * w_out * x.dtype().size_in_bytes();
    if k == 1 || cfg.stride != 1 || cfg.dilation != 1 || cfg.groups != 1 || row_bytes * h_out <= budget_bytes {
        return x.apply(conv);
    }
    let padded = x.pad_with_zeros(2, p, p)?.pad_with_zeros(3, p, p)?;
    let band = (budget_bytes / row_bytes).max(1);
    let mut outs = Vec::with_capacity(h_out.div_ceil(band));
    for r0 in (0..h_out).step_by(band) {
        let rows = band.min(h_out - r0);
        let y = padded.narrow(2, r0, rows + k - 1)?.contiguous()?.conv2d(conv.weight(), 0, 1, 1, 1)?;
        outs.push(match conv.bias() {
            Some(b) => y.broadcast_add(&b.reshape((1, (), 1, 1))?)?,
            None => y,
        });
    }
    Tensor::cat(&outs, 2)
}

/// Residual block: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2, plus a
/// 1x1 shortcut conv when channel count changes. Matches `QwenImage21ResidualBlock`.
/// The native checkpoint stores this as an indexed `nn.Sequential` (`residual.0`
/// = norm1, `.2` = conv1, `.3` = norm2, `.6` = conv2, sibling `shortcut`); the
/// diffusers checkpoint uses plain names (`norm1`/`conv1`/`norm2`/`conv2`/`conv_shortcut`).
#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: RmsNormChannelFirst,
    conv1: candle_nn::Conv2d,
    norm2: RmsNormChannelFirst,
    conv2: candle_nn::Conv2d,
    conv_shortcut: Option<candle_nn::Conv2d>,
}

impl ResidualBlock {
    fn new(in_dim: usize, out_dim: usize, native: bool, vb: VarBuilder) -> Result<Self> {
        let (p_norm1, p_conv1, p_norm2, p_conv2, p_shortcut) = if native {
            ("residual.0", "residual.2", "residual.3", "residual.6", "shortcut")
        } else {
            ("norm1", "conv1", "norm2", "conv2", "conv_shortcut")
        };
        let norm1 = RmsNormChannelFirst::new(in_dim, 3, vb.pp(p_norm1))?;
        let conv1 = load_conv(native, in_dim, out_dim, 3, 1, vb.pp(p_conv1))?;
        let norm2 = RmsNormChannelFirst::new(out_dim, 3, vb.pp(p_norm2))?;
        let conv2 = load_conv(native, out_dim, out_dim, 3, 1, vb.pp(p_conv2))?;
        let conv_shortcut = if in_dim != out_dim {
            Some(load_conv(native, in_dim, out_dim, 1, 0, vb.pp(p_shortcut))?)
        } else {
            None
        };
        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = match &self.conv_shortcut {
            Some(c) => x.apply(c)?,
            None => x.clone(),
        };
        let mut y = self.norm1.forward(x)?;
        y = y.silu()?;
        y = conv2d_banded(&y, &self.conv1)?;
        y = self.norm2.forward(&y)?;
        y = y.silu()?;
        y = conv2d_banded(&y, &self.conv2)?;
        y + h
    }
}

/// Single-head causal self-attention over spatial positions. Matches
/// `QwenImage21AttentionBlock` (used once, inside the mid-block).
#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: RmsNormChannelFirst,
    to_qkv: candle_nn::Conv2d,
    proj: candle_nn::Conv2d,
}

impl AttentionBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = RmsNormChannelFirst::new(dim, 2, vb.pp("norm"))?;
        let to_qkv = candle_nn::conv2d(dim, dim * 3, 1, Default::default(), vb.pp("to_qkv"))?;
        let proj = candle_nn::conv2d(dim, dim, 1, Default::default(), vb.pp("proj"))?;
        Ok(Self { norm, to_qkv, proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let (b, c, h, w) = x.dims4()?;

        let xn = self.norm.forward(x)?;
        let qkv = xn.apply(&self.to_qkv)?; // [b, 3c, h, w]
        let qkv = qkv.reshape((b, 3 * c, h * w))?.transpose(1, 2)?; // [b, hw, 3c]
        let q = qkv.narrow(2, 0, c)?;
        let k = qkv.narrow(2, c, c)?;
        let v = qkv.narrow(2, 2 * c, c)?;

        let scale = (c as f64).powf(-0.5);
        let attn = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?; // [b, hw, c]
        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?;
        let out = out.apply(&self.proj)?;
        out + identity
    }
}

/// Mid-block: resnet -> attention -> resnet (`num_layers=1`). Matches `QwenImage21MidBlock`.
#[derive(Debug, Clone)]
struct MidBlock {
    resnet0: ResidualBlock,
    attn: AttentionBlock,
    resnet1: ResidualBlock,
}

impl MidBlock {
    fn new(dim: usize, native: bool, vb: VarBuilder) -> Result<Self> {
        let (p0, p1, p2) = if native { ("0", "1", "2") } else { ("resnets.0", "attentions.0", "resnets.1") };
        let resnet0 = ResidualBlock::new(dim, dim, native, vb.pp(p0))?;
        let attn = AttentionBlock::new(dim, vb.pp(p1))?;
        let resnet1 = ResidualBlock::new(dim, dim, native, vb.pp(p2))?;
        Ok(Self { resnet0, attn, resnet1 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }
}

/// Nearest-neighbor depth-to-space upsample via channel repeat, matching
/// `QwenImage21DupUp3D` (with `first_chunk=True`, the only case that applies to
/// single-frame image decoding). `factor_t` still affects the result whenever
/// `in_channels != out_channels`, since it changes the repeat count and thus
/// which input channel maps to which output position — it is not a no-op.
fn dup_up3d(x: &Tensor, out_channels: usize, factor_t: usize, factor_s: usize) -> Result<Tensor> {
    let (b, c_in, h, w) = x.dims4()?;
    let factor = factor_t * factor_s * factor_s;
    let repeats = out_channels * factor / c_in;

    let x = x.unsqueeze(2)?.broadcast_as((b, c_in, repeats, h, w))?.contiguous()?;
    let x = x.reshape((b, c_in * repeats, h, w))?; // == out_channels * factor

    // Split channels into (out, factor_t, fs*fs) and keep only the last temporal
    // sub-slice (matching `first_chunk=True`) before doing the spatial split, so
    // no intermediate tensor needs more than 6 dims.
    let fs2 = factor_s * factor_s;
    let x = x.reshape((b, out_channels, factor_t, fs2, h, w))?;
    let x = x.narrow(2, factor_t - 1, 1)?.squeeze(2)?; // [b, out, fs2, h, w]

    let x = x.reshape((b, out_channels, factor_s, factor_s, h, w))?;
    let x = x.permute((0, 1, 4, 2, 5, 3))?.contiguous()?; // [b, out, h, fs, w, fs]
    x.reshape((b, out_channels, h * factor_s, w * factor_s))
}

/// Decoder up-block: resnets, then (if upsampling) a nearest+conv resample plus
/// a parallel depth-to-space shortcut added on top. Matches `QwenImage21ResidualUpBlock`
/// (the `is_residual=true` path, which is what this checkpoint uses).
#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResidualBlock>,
    upsampler: Option<candle_nn::Conv2d>,
    avg_shortcut_factor_t: Option<usize>,
    out_dim: usize,
}

impl UpBlock {
    fn new(
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        native: bool,
        up_flag: bool,
        temporal_upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks + 1);
        let mut current = in_dim;
        for i in 0..=num_res_blocks {
            let p = if native { format!("{}", i) } else { format!("resnets.{}", i) };
            resnets.push(ResidualBlock::new(current, out_dim, native, vb.pp(p))?);
            current = out_dim;
        }

        let (upsampler, avg_shortcut_factor_t) = if up_flag {
            // The resample conv is a genuine 2D conv (4D weight) in both layouts,
            // unlike the CausalConv3d-backed resnets above.
            let p = if native {
                format!("{}.resample.1", num_res_blocks + 1)
            } else {
                "upsampler.resample.1".to_string()
            };
            let conv = candle_nn::conv2d(
                out_dim, out_dim, 3,
                Conv2dConfig { padding: 1, ..Default::default() },
                vb.pp(p),
            )?;
            let factor_t = if temporal_upsample { 2 } else { 1 };
            (Some(conv), Some(factor_t))
        } else {
            (None, None)
        };

        Ok(Self { resnets, upsampler, avg_shortcut_factor_t, out_dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_copy = x.clone();
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        if let Some(conv) = &self.upsampler {
            let h = x.dim(D::Minus2)? * 2;
            let w = x.dim(D::Minus1)? * 2;
            x = x.upsample_nearest2d(h, w)?;
            x = conv2d_banded(&x, conv)?;
        }
        if let Some(factor_t) = self.avg_shortcut_factor_t {
            let shortcut = dup_up3d(&x_copy, self.out_dim, factor_t, 2)?;
            x = (x + shortcut)?;
        }
        Ok(x)
    }
}

/// VAE Decoder (`AutoencoderKLQwenImage21`, decode-only — encoding is not needed
/// since this pipeline only ever decodes transformer output back to pixels).
#[derive(Debug, Clone)]
pub struct VaeDecoder {
    post_quant_conv: candle_nn::Conv2d,
    conv_in: candle_nn::Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: RmsNormChannelFirst,
    conv_out: candle_nn::Conv2d,
}

impl VaeDecoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.decoder_base_dim;
        let last_mult = *cfg.dim_mult.last().unwrap();
        let mut mults = vec![last_mult];
        let mut rev = cfg.dim_mult.clone();
        rev.reverse();
        mults.extend(rev);
        let dims: Vec<usize> = mults.iter().map(|&m| dim * m).collect();

        // Two checkpoint distributions exist for this VAE, with identical layer
        // graphs (channel dims, block counts) but different tensor names and
        // conv-weight ranks — see `load_conv`. "native" is the ComfyUI-repackaged
        // layout (`conv2`/`decoder.middle`/`decoder.upsamples.<i>.upsamples.<j>`,
        // 5D conv weights); otherwise this is the official diffusers `vae/`
        // checkpoint (`post_quant_conv`/`decoder.mid_block`/`decoder.up_blocks.<i>`,
        // plain 4D conv weights).
        let native = vb.contains_tensor("conv2.weight");
        let (p_post_quant_conv, p_conv_in, p_mid_block, p_norm_out, p_conv_out) = if native {
            ("conv2", "decoder.conv1", "decoder.middle", "decoder.head.0", "decoder.head.2")
        } else {
            ("post_quant_conv", "decoder.conv_in", "decoder.mid_block", "decoder.norm_out", "decoder.conv_out")
        };

        let post_quant_conv = load_conv(native, cfg.z_dim, cfg.z_dim, 1, 0, vb.pp(p_post_quant_conv))?;
        let conv_in = load_conv(native, cfg.z_dim, dims[0], 3, 1, vb.pp(p_conv_in))?;
        let mid_block = MidBlock::new(dims[0], native, vb.pp(p_mid_block))?;

        let mut temporal_upsample = cfg.temperal_downsample.clone();
        temporal_upsample.reverse();
        let num_transitions = cfg.dim_mult.len();

        let mut up_blocks = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            let up_flag = i != num_transitions - 1;
            let temporal = up_flag && temporal_upsample.get(i).copied().unwrap_or(false);
            let p = if native {
                format!("decoder.upsamples.{}.upsamples", i)
            } else {
                format!("decoder.up_blocks.{}", i)
            };
            let block = UpBlock::new(
                dims[i], dims[i + 1], cfg.num_res_blocks, native, up_flag, temporal,
                vb.pp(p),
            )?;
            up_blocks.push(block);
        }

        // Native: `head` is a 3-element Sequential ([0]=norm RMSNorm, [1]=SiLU, [2]=conv_out).
        // Diffusers: plain `norm_out`/`conv_out` names.
        let out_dim = *dims.last().unwrap();
        let norm_out = RmsNormChannelFirst::new(out_dim, 3, vb.pp(p_norm_out))?;
        let conv_out = load_conv(native, out_dim, cfg.out_channels, 3, 1, vb.pp(p_conv_out))?;

        Ok(Self { post_quant_conv, conv_in, mid_block, up_blocks, norm_out, conv_out })
    }

    /// Decode latents `[B, z_dim, H, W]` to pixels `[B, out_channels, H*16, W*16]`.
    ///
    /// Everything after the mid-block (whose attention is global) is spatially
    /// local, so each up-block runs over row bands (see [`in_row_bands`]): the
    /// full-resolution intermediates never exist for the whole image at once.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut x = latents.apply(&self.post_quant_conv)?;
        x = conv2d_banded(&x, &self.conv_in)?;
        x = self.mid_block.forward(&x)?;
        let last = self.up_blocks.len() - 1;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let scale = if up_block.upsampler.is_some() { 2 } else { 1 };
            let (_, c_in, _, w_in) = x.dims4()?;
            let bytes_per_row = c_in.max(up_block.out_dim) * w_in * scale * scale * x.dtype().size_in_bytes();
            let band = (UP_BAND_BUDGET_BYTES / bytes_per_row).max(1);
            x = in_row_bands(&x, UP_BLOCK_HALO_ROWS, scale, band, |band_x| {
                let y = up_block.forward(band_x)?;
                if i == last { conv2d_banded(&self.norm_out.forward(&y)?.silu()?, &self.conv_out) } else { Ok(y) }
            })?;
        }
        x.clamp(-1.0f64, 1.0f64)
    }
}

/// Receptive radius of one up-block, in its input rows: 3 residual blocks of
/// two 3x3 convs (6 rows), plus the resample conv after the 2x upsample (half
/// a row) or, for the last block, the 3x3 `conv_out` head (1 row).
const UP_BLOCK_HALO_ROWS: usize = 7;

/// Target size of one up-block band's largest tensor.
const UP_BAND_BUDGET_BYTES: usize = 256 << 20;

/// `f(x)` for a spatially local `f` whose output has `scale`x the input's rows
/// and a receptive radius of at most `halo` input rows, computed over
/// horizontal bands of `band` rows (each extended by `halo` rows on both sides
/// where the image continues). Exact: cutting the input only perturbs output
/// rows within the halo, which are dropped; at the true image edges `f` sees
/// the same zero padding as on the whole image. Bands are written into one
/// preallocated output, so only one band's intermediates are alive at a time.
fn in_row_bands(x: &Tensor, halo: usize, scale: usize, band: usize, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Tensor> {
    let h = x.dim(2)?;
    if band >= h {
        return f(x);
    }
    let mut out: Option<Tensor> = None;
    for r0 in (0..h).step_by(band) {
        let r1 = (r0 + band).min(h);
        let (s0, s1) = (r0.saturating_sub(halo), (r1 + halo).min(h));
        let y = f(&x.narrow(2, s0, s1 - s0)?)?;
        let piece = y.narrow(2, (r0 - s0) * scale, (r1 - r0) * scale)?.contiguous()?;
        let out = match &out {
            Some(o) => o,
            None => {
                let (b, c, _, w) = piece.dims4()?;
                out.insert(Tensor::zeros((b, c, h * scale, w), piece.dtype(), piece.device())?)
            }
        };
        out.slice_set(&piece, 2, r0 * scale)?;
    }
    Ok(out.expect("h > band >= 1, so at least one band ran"))
}

/// `QwenImage21AvgDown3D` for a single frame: space-to-depth by `factor_s`,
/// then channels averaged in groups down to `out_channels`. With `factor_t=2`
/// the upstream module front-pads the one-frame input with a zero frame, so the
/// zero frame fills temporal slot 0 and is averaged in too — replicated here,
/// since skipping it would change every temporally-downsampling block's shortcut.
fn avg_down3d(x: &Tensor, out_channels: usize, factor_t: usize, factor_s: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let (ho, wo, fs2) = (h / factor_s, w / factor_s, factor_s * factor_s);
    // [B, C, fs, fs, H', W'] -> channel order (c, sh, sw), as upstream's permute.
    let s2d = x
        .reshape((b, c, ho, factor_s, wo, factor_s))?
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c, 1, fs2, ho, wo))?;
    let y = match factor_t {
        1 => s2d,
        2 => Tensor::cat(&[&s2d.zeros_like()?, &s2d], 2)?,
        _ => candle_core::bail!("avg_down3d: unsupported factor_t {factor_t}"),
    };
    let group = c * factor_t * fs2 / out_channels;
    y.reshape((b, out_channels, group, ho, wo))?.mean(2)
}

/// Encoder down-block (`QwenImage21ResidualDownBlock`, `is_residual=true`):
/// resnets, then an optional downsampler (right/bottom zero-pad by 1 + stride-2
/// 3x3 conv), plus the parallel `avg_down3d` shortcut of the block input. The
/// downsampler's `time_conv` only runs from the second frame on, so it is
/// unused for single images.
#[derive(Debug, Clone)]
struct DownBlock {
    resnets: Vec<ResidualBlock>,
    downsample: Option<candle_nn::Conv2d>,
    out_dim: usize,
    factor_t: usize,
    factor_s: usize,
}

impl DownBlock {
    fn new(in_dim: usize, out_dim: usize, num_res_blocks: usize, down_flag: bool, temporal: bool, vb: VarBuilder) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks);
        let mut dim = in_dim;
        for j in 0..num_res_blocks {
            resnets.push(ResidualBlock::new(dim, out_dim, false, vb.pp(format!("resnets.{j}")))?);
            dim = out_dim;
        }
        let downsample = if down_flag {
            let cfg = Conv2dConfig { stride: 2, ..Default::default() };
            Some(candle_nn::conv2d(out_dim, out_dim, 3, cfg, vb.pp("downsampler.resample.1"))?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsample,
            out_dim,
            factor_t: if temporal { 2 } else { 1 },
            factor_s: if down_flag { 2 } else { 1 },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.clone();
        for resnet in &self.resnets {
            y = resnet.forward(&y)?;
        }
        if let Some(conv) = &self.downsample {
            y = y.pad_with_zeros(D::Minus1, 0, 1)?.pad_with_zeros(D::Minus2, 0, 1)?.apply(conv)?;
        }
        y + avg_down3d(x, self.out_dim, self.factor_t, self.factor_s)?
    }
}

/// Encoder of `AutoencoderKLQwenImage21` (official diffusers layout only), for
/// turning condition images into latents.
#[derive(Debug, Clone)]
pub struct VaeEncoder {
    conv_in: candle_nn::Conv2d,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    norm_out: RmsNormChannelFirst,
    conv_out: candle_nn::Conv2d,
    quant_conv: candle_nn::Conv2d,
    z_dim: usize,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl VaeEncoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut dims = vec![cfg.encoder_base_dim];
        dims.extend(cfg.dim_mult.iter().map(|&m| cfg.encoder_base_dim * m));
        let n = cfg.dim_mult.len();
        let enc = vb.pp("encoder");
        let conv_in = candle_nn::conv2d(cfg.in_channels, dims[0], 3, Conv2dConfig { padding: 1, ..Default::default() }, enc.pp("conv_in"))?;
        let mut down_blocks = Vec::with_capacity(n);
        for i in 0..n {
            let down_flag = i != n - 1;
            let temporal = down_flag && cfg.temperal_downsample.get(i).copied().unwrap_or(false);
            down_blocks.push(DownBlock::new(dims[i], dims[i + 1], cfg.num_res_blocks, down_flag, temporal, enc.pp(format!("down_blocks.{i}")))?);
        }
        let out_dim = dims[n];
        let mid_block = MidBlock::new(out_dim, false, enc.pp("mid_block"))?;
        let norm_out = RmsNormChannelFirst::new(out_dim, 3, enc.pp("norm_out"))?;
        let conv_out = candle_nn::conv2d(out_dim, 2 * cfg.z_dim, 3, Conv2dConfig { padding: 1, ..Default::default() }, enc.pp("conv_out"))?;
        let quant_conv = candle_nn::conv2d(2 * cfg.z_dim, 2 * cfg.z_dim, 1, Default::default(), vb.pp("quant_conv"))?;
        Ok(Self {
            conv_in, down_blocks, mid_block, norm_out, conv_out, quant_conv,
            z_dim: cfg.z_dim,
            latents_mean: cfg.latents_mean.clone(),
            latents_std: cfg.latents_std.clone(),
        })
    }

    /// Encode an RGBA image `[B, 4, H, W]` in `[-1, 1]` into normalized latents
    /// `[B, z_dim, H/16, W/16]`: the posterior mean (upstream's
    /// `sample_mode="argmax"`), then `(x - latents_mean) / latents_std`.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let mut x = conv2d_banded(image, &self.conv_in)?;
        for block in &self.down_blocks {
            x = block.forward(&x)?;
        }
        x = self.mid_block.forward(&x)?;
        x = conv2d_banded(&self.norm_out.forward(&x)?.silu()?, &self.conv_out)?.apply(&self.quant_conv)?;
        let mean = x.narrow(1, 0, self.z_dim)?;
        let (device, dt) = (mean.device(), mean.dtype());
        let m = Tensor::new(self.latents_mean.as_slice(), device)?.to_dtype(dt)?.reshape((1, self.z_dim, 1, 1))?;
        let s = Tensor::new(self.latents_std.as_slice(), device)?.to_dtype(dt)?.reshape((1, self.z_dim, 1, 1))?;
        mean.broadcast_sub(&m)?.broadcast_div(&s)
    }
}

/// Unpack a transformer token sequence `[B, H*W, C]` back into `[B, C, 1, H, W]`.
/// Qwen-Image-2.1's transformer uses `patch_size=1` (each latent pixel is its own
/// token, `in_channels`/`out_channels` == the VAE's `z_dim` directly) — unlike
/// Flux/SD3-style models that pack 2x2 spatial patches into 4x the channel count.
pub fn unpack_latents(
    latents: &Tensor,
    height: usize,
    width: usize,
    vae_scale_factor: usize,
) -> Result<Tensor> {
    let (b, _num_tokens, c) = latents.dims3()?;
    let h = height / vae_scale_factor;
    let w = width / vae_scale_factor;

    let x = latents.transpose(1, 2)?.contiguous()?; // [B, C, H*W]
    let x = x.reshape((b, c, h, w))?;
    x.unsqueeze(2) // [B, C, 1, H, W]
}

/// Flatten latents `[B, C, 1, H, W]` into a transformer token sequence `[B, H*W, C]`.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let (b, c, _t, h, w) = latents.dims5()?;
    let x = latents.squeeze(2)?; // [B, C, H, W]
    let x = x.reshape((b, c, h * w))?;
    x.transpose(1, 2)?.contiguous() // [B, H*W, C]
}

/// Normalize latents using mean and std.
pub fn normalize_latents(latents: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
    let device = latents.device();
    let dt = latents.dtype();

    let c = mean.len();
    let mean_t = Tensor::new(mean, device)?.to_dtype(dt)?.reshape((1, c, 1, 1, 1))?;
    let inv_std: Vec<f32> = std.iter().map(|s| 1.0 / s).collect();
    let std_t = Tensor::new(inv_std.as_slice(), device)?.to_dtype(dt)?.reshape((1, c, 1, 1, 1))?;

    let x = latents.broadcast_div(&std_t)?;
    x.broadcast_add(&mean_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    /// Row-banded evaluation of a local op must equal evaluating it whole, for
    /// bands down to a single row and a final band shorter than the others.
    #[test]
    fn test_in_row_bands_matches_whole() {
        // A local op with radius 2 input rows and 2x row upsampling: two 3x3
        // convs around a nearest upsample (radius 1 + 1/2 -> rounded to 2).
        let dev = Device::Cpu;
        let w1 = Tensor::randn(0f32, 1.0, (3, 2, 3, 3), &dev).unwrap();
        let w2 = Tensor::randn(0f32, 1.0, (2, 3, 3, 3), &dev).unwrap();
        let f = |x: &Tensor| -> Result<Tensor> {
            let y = x.conv2d(&w1, 1, 1, 1, 1)?.silu()?;
            let (h, w) = (y.dim(2)? * 2, y.dim(3)? * 2);
            y.upsample_nearest2d(h, w)?.conv2d(&w2, 1, 1, 1, 1)
        };
        let x = Tensor::randn(0f32, 1.0, (1, 2, 13, 5), &dev).unwrap();
        let want = f(&x).unwrap();
        for band in [1, 3, 4, 12] {
            let got = in_row_bands(&x, 2, 2, band, f).unwrap();
            let diff = (&got - &want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(diff < 1e-5, "band {band}: max diff {diff}");
        }
    }

    /// Banded convolution must equal the direct one (same weights and padding),
    /// including a final band shorter than the others.
    #[test]
    fn test_conv2d_banded_matches_direct() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (5, 3, 3, 3), &device).unwrap();
        let bias = Tensor::randn(0f32, 1f32, 5, &device).unwrap();
        let conv = candle_nn::Conv2d::new(weight, Some(bias), Conv2dConfig { padding: 1, ..Default::default() });
        let x = Tensor::randn(0f32, 1f32, (1, 3, 23, 17), &device).unwrap();
        // 3·3·3·17·4 = 1836 bytes per output row; a 7-row budget gives bands of 7,7,7,2.
        let banded = conv2d_banded_with_budget(&x, &conv, 1836 * 7).unwrap();
        let direct = x.apply(&conv).unwrap();
        assert_eq!(banded.dims(), direct.dims());
        let diff = (direct - banded).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-5, "max diff {diff}");
    }

    /// Builds against a small config so the smoke test runs fast on CPU, but keeps
    /// the same dim_mult/z_dim ratios; this exercises tensor-name wiring and the
    /// channel/spatial arithmetic through every up_block (equal-channel, channel-
    /// reducing, and the final no-upsample block) without needing a real checkpoint.
    #[test]
    fn test_decoder_shapes() {
        let device = Device::Cpu;
        let cfg = Config {
            z_dim: 8,
            encoder_base_dim: 4,
            in_channels: 4,
            decoder_base_dim: 4,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 1,
            temperal_downsample: vec![false, true, true, true],
            out_channels: 4,
            scale_factor_spatial: 16,
            latents_mean: vec![0.0; 8],
            latents_std: vec![1.0; 8],
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let decoder = VaeDecoder::new(&cfg, vb).expect("decoder should build with matching tensor names");

        let latents = Tensor::zeros((1, cfg.z_dim, 3, 5), DType::F32, &device).unwrap();
        let out = decoder.decode(&latents).unwrap();
        assert_eq!(out.dims(), &[1, cfg.out_channels, 3 * 16, 5 * 16]);
    }

    /// Cross-checks `dup_up3d` against the literal diffusers `QwenImage21DupUp3D.forward`
    /// formula (repeat_interleave -> 8D view -> permute(0,1,5,2,6,3,7,4) -> merge ->
    /// keep index [factor_t-1:]), hand-traced for a small, fully deterministic case
    /// where numpy/torch weren't available to run directly: in_channels=2,
    /// out_channels=4, factor_t=2, factor_s=2, H=W=2. Input channel 0 is [[0,1],[2,3]],
    /// channel 1 is [[4,5],[6,7]]. Tracing the real formula's index arithmetic by hand
    /// shows repeats=16 puts output channels {0,1} both fed from input channel 0 and
    /// {2,3} both fed from channel 1 — i.e. each input channel's 2x2 grid is plainly
    /// nearest-upsampled to 4x4 and duplicated across two adjacent output channels
    /// (see git history / PR discussion for the full derivation).
    #[test]
    fn test_dup_up3d_matches_hand_traced_reference() {
        let device = Device::Cpu;
        #[rustfmt::skip]
        let input = Tensor::new(
            &[
                0.0f32, 1.0, 2.0, 3.0, // channel 0: [[0,1],[2,3]]
                4.0, 5.0, 6.0, 7.0,    // channel 1: [[4,5],[6,7]]
            ],
            &device,
        )
        .unwrap()
        .reshape((1, 2, 2, 2))
        .unwrap();

        let out = dup_up3d(&input, 4, 2, 2).unwrap();
        assert_eq!(out.dims(), &[1, 4, 4, 4]);

        let up = |block: &[f32; 4]| -> Vec<f32> {
            // Plain 2x nearest upsample of a 2x2 block to 4x4, row-major.
            let (a, b, c, d) = (block[0], block[1], block[2], block[3]);
            vec![
                a, a, b, b,
                a, a, b, b,
                c, c, d, d,
                c, c, d, d,
            ]
        };
        let expected: Vec<f32> = [
            up(&[0.0, 1.0, 2.0, 3.0]), // out channel 0 <- input channel 0
            up(&[0.0, 1.0, 2.0, 3.0]), // out channel 1 <- input channel 0 (duplicated)
            up(&[4.0, 5.0, 6.0, 7.0]), // out channel 2 <- input channel 1
            up(&[4.0, 5.0, 6.0, 7.0]), // out channel 3 <- input channel 1 (duplicated)
        ]
        .concat();

        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, expected);
    }
}
