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
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
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

/// Residual block: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2, plus a
/// 1x1 shortcut conv when channel count changes. Matches `QwenImage21ResidualBlock`.
#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: RmsNormChannelFirst,
    conv1: candle_nn::Conv2d,
    norm2: RmsNormChannelFirst,
    conv2: candle_nn::Conv2d,
    conv_shortcut: Option<candle_nn::Conv2d>,
}

impl ResidualBlock {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = RmsNormChannelFirst::new(in_dim, vb.pp("norm1"))?;
        let conv1 = candle_nn::conv2d(
            in_dim, out_dim, 3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv1"),
        )?;
        let norm2 = RmsNormChannelFirst::new(out_dim, vb.pp("norm2"))?;
        let conv2 = candle_nn::conv2d(
            out_dim, out_dim, 3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv2"),
        )?;
        let conv_shortcut = if in_dim != out_dim {
            Some(candle_nn::conv2d(in_dim, out_dim, 1, Default::default(), vb.pp("conv_shortcut"))?)
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
        y = y.apply(&self.conv1)?;
        y = self.norm2.forward(&y)?;
        y = y.silu()?;
        y = y.apply(&self.conv2)?;
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
        let norm = RmsNormChannelFirst::new(dim, vb.pp("norm"))?;
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
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let resnet0 = ResidualBlock::new(dim, dim, vb.pp("resnets.0"))?;
        let attn = AttentionBlock::new(dim, vb.pp("attentions.0"))?;
        let resnet1 = ResidualBlock::new(dim, dim, vb.pp("resnets.1"))?;
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
        up_flag: bool,
        temporal_upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks + 1);
        let mut current = in_dim;
        for i in 0..=num_res_blocks {
            resnets.push(ResidualBlock::new(current, out_dim, vb.pp(format!("resnets.{}", i)))?);
            current = out_dim;
        }

        let (upsampler, avg_shortcut_factor_t) = if up_flag {
            let conv = candle_nn::conv2d(
                out_dim, out_dim, 3,
                Conv2dConfig { padding: 1, ..Default::default() },
                vb.pp("upsampler.resample.1"),
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
            x = x.apply(conv)?;
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

        let post_quant_conv = candle_nn::conv2d(cfg.z_dim, cfg.z_dim, 1, Default::default(), vb.pp("post_quant_conv"))?;
        let conv_in = candle_nn::conv2d(
            cfg.z_dim, dims[0], 3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("decoder.conv_in"),
        )?;
        let mid_block = MidBlock::new(dims[0], vb.pp("decoder.mid_block"))?;

        let mut temporal_upsample = cfg.temperal_downsample.clone();
        temporal_upsample.reverse();
        let num_transitions = cfg.dim_mult.len();

        let mut up_blocks = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            let up_flag = i != num_transitions - 1;
            let temporal = up_flag && temporal_upsample.get(i).copied().unwrap_or(false);
            let block = UpBlock::new(
                dims[i], dims[i + 1], cfg.num_res_blocks, up_flag, temporal,
                vb.pp(format!("decoder.up_blocks.{}", i)),
            )?;
            up_blocks.push(block);
        }

        let out_dim = *dims.last().unwrap();
        let norm_out = RmsNormChannelFirst::new(out_dim, vb.pp("decoder.norm_out"))?;
        let conv_out = candle_nn::conv2d(
            out_dim, cfg.out_channels, 3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("decoder.conv_out"),
        )?;

        Ok(Self { post_quant_conv, conv_in, mid_block, up_blocks, norm_out, conv_out })
    }

    /// Decode latents `[B, z_dim, H, W]` to pixels `[B, out_channels, H*16, W*16]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut x = latents.apply(&self.post_quant_conv)?;
        x = x.apply(&self.conv_in)?;
        x = self.mid_block.forward(&x)?;
        for up_block in &self.up_blocks {
            x = up_block.forward(&x)?;
        }
        x = self.norm_out.forward(&x)?;
        x = x.silu()?;
        x = x.apply(&self.conv_out)?;
        x.clamp(-1.0f64, 1.0f64)
    }
}

/// Unpack latents back to image format.
pub fn unpack_latents(
    latents: &Tensor,
    height: usize,
    width: usize,
    vae_scale_factor: usize,
) -> Result<Tensor> {
    let (b, _num_patches, channels_packed) = latents.dims3()?;
    let channels = channels_packed / 4;
    let patch_size = 2;

    let h = 2 * (height / (vae_scale_factor * 2));
    let w = 2 * (width / (vae_scale_factor * 2));

    let x = latents.reshape((b, h / 2, w / 2, channels, patch_size, patch_size))?;
    let x = x.permute((0, 3, 1, 4, 2, 5))?;
    let x = x.reshape((b, channels, 1, h, w))?;

    Ok(x)
}

/// Pack latents for transformer input.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let (_b, _c, _t, h, w) = latents.dims5()?;
    let p = 2;

    let x = latents.squeeze(2)?;

    let pad_h = (p - h % p) % p;
    let pad_w = (p - w % p) % p;
    let x = if pad_h > 0 {
        x.pad_with_zeros(D::Minus2, 0, pad_h)?
    } else {
        x
    };
    let x = if pad_w > 0 {
        x.pad_with_zeros(D::Minus1, 0, pad_w)?
    } else {
        x
    };

    let (b, c, h_pad, w_pad) = x.dims4()?;
    let h_patches = h_pad / p;
    let w_patches = w_pad / p;

    let x = x.reshape((b, c, h_patches, 2, w_patches, 2))?;
    let x = x.permute((0, 3, 4, 5, 1, 2))?;
    x.reshape((b, h_patches * w_patches, c * 4))
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

    /// Builds against a small config so the smoke test runs fast on CPU, but keeps
    /// the same dim_mult/z_dim ratios; this exercises tensor-name wiring and the
    /// channel/spatial arithmetic through every up_block (equal-channel, channel-
    /// reducing, and the final no-upsample block) without needing a real checkpoint.
    #[test]
    fn test_decoder_shapes() {
        let device = Device::Cpu;
        let cfg = Config {
            z_dim: 8,
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
}
