use candle_core::{Result, Tensor, Device, DType};
use crate::transformer::QwenImageTransformer;
use crate::quantized_transformer::QwenImageTransformerQuantized;
use crate::vae::{Config as VaeConfig, VaeDecoder, pack_latents, unpack_latents, normalize_latents};

/// Transformer type enum for supporting both quantized and non-quantized modes.
#[derive(Debug, Clone)]
pub enum TransformerType {
    NonQuantized(QwenImageTransformer),
    Quantized(QwenImageTransformerQuantized),
}

impl TransformerType {
    /// Run the transformer forward pass.
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        text_emb: &Tensor,
        text_emb_mask: Option<&Tensor>,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        eprintln!("  [TransformerType::forward] variant: {:?}", match self { TransformerType::NonQuantized(_) => "NonQuantized", TransformerType::Quantized(_) => "Quantized" });
        match self {
            Self::NonQuantized(t) => t.forward(latents, Some(text_emb), text_emb_mask, timestep, img_ids, txt_ids, height, width),
            Self::Quantized(t) => t.forward(latents, Some(text_emb), text_emb_mask, timestep, img_ids, txt_ids, height, width),
        }
    }

    /// Number of transformer blocks (for KV cache allocation).
    pub fn num_blocks(&self) -> usize {
        match self {
            Self::NonQuantized(t) => t.transformer_blocks.len(),
            Self::Quantized(t) => t.transformer_blocks.len(),
        }
    }
}

/// Lightweight pipeline config — holds no model instances.
/// Models are loaded/dropped by the caller in phases to control memory.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub vae_cfg: VaeConfig,
    pub device: Device,
    pub dtype: DType,
}

/// Run the denoising loop. Returns packed latents.
///
/// Caller owns the transformer lifecycle — drop it after this returns
/// to free memory before loading the VAE.
pub fn denoise(
    transformer: &TransformerType,
    prompt_emb: &Tensor,
    height: usize,
    width: usize,
    num_inference_steps: usize,
    cfg: &PipelineConfig,
) -> Result<Tensor> {
    let batch_size = prompt_emb.dim(0)?;
    let num_channels_latents = cfg.vae_cfg.z_dim;
    let vae_scale_factor = cfg.vae_cfg.spatial_compression_ratio();
    let latent_h = height / vae_scale_factor;
    let latent_w = width / vae_scale_factor;

    // Prepare random latents: [B, C, T, H, W] (temporal at index 2)
    let shape = (batch_size, num_channels_latents, 1, latent_h, latent_w);
    let latents = Tensor::randn(0.0f64, 1.0f64, shape, &cfg.device)?;
    let mut packed_latents = pack_latents(&latents)?.to_dtype(cfg.dtype)?;
    let prompt_emb = prompt_emb.to_dtype(cfg.dtype)?;

    // Scheduler: linear sigma schedule
    let sigmas: Vec<f32> = (0..num_inference_steps)
        .map(|i| 1.0 - (i as f32) / (num_inference_steps as f32))
        .collect();
    let timesteps: Vec<f32> = (0..num_inference_steps)
        .map(|i| sigmas[i])
        .collect();

    // Prepare RoPE position IDs
    let h_patches = height / vae_scale_factor;
    let w_patches = width / vae_scale_factor;
    let img_ids = crate::rope::compute_img_ids(1, h_patches, w_patches, &cfg.device)?;
    let txt_ids = crate::rope::compute_txt_ids(1, prompt_emb.dim(1)?, 0, &cfg.device)?;

    eprintln!("  Initial packed_latents shape: {:?}", packed_latents.shape());
    eprintln!("  prompt_emb shape: {:?}", prompt_emb.shape());
    eprintln!("  img_ids shape: {:?}", img_ids.shape());
    eprintln!("  txt_ids shape: {:?}", txt_ids.shape());

    // Denoising loop
    for i in 0..num_inference_steps {
        let t = timesteps[i] as f64;
        // [B] (batch=1); `timestep_embedding` does its own unsqueeze to broadcast
        // against the exponent table, so this must stay 1-D here.
        let timestep = Tensor::new(&[t], &cfg.device)?
            .to_dtype(packed_latents.dtype())?;

        // Apply timestep embedding (sinusoidal -> 256 dim)
        let timestep = crate::scheduler::timestep_embedding(&timestep, 256)?
            .to_dtype(packed_latents.dtype())?;

        let noise_pred = transformer.forward(
            &packed_latents, &timestep, &prompt_emb, None, &img_ids, &txt_ids,
            height, width,
        )?;

        if i == 0 {
            eprintln!("  noise_pred shape: {:?}", noise_pred.shape());
        }

        // Euler step: x_{t-1} = x_t + (sigma_{t+1} - sigma_t) * noise_pred
        let dt = if i + 1 < sigmas.len() {
            sigmas[i + 1] - sigmas[i]
        } else {
            -sigmas[i]
        };
        let dt_tensor = Tensor::new(&[dt], &cfg.device)?.to_dtype(packed_latents.dtype())?;
        packed_latents = packed_latents.broadcast_add(&(noise_pred.broadcast_mul(&dt_tensor)?))?;

        if i % 10 == 0 {
            eprintln!("  Step {}/{}", i + 1, num_inference_steps);
        }
    }

    eprintln!("  Final packed_latents shape: {:?}", packed_latents.shape());
    Ok(packed_latents)
}

/// Decode packed latents to image. Caller owns the VAE lifecycle.
pub fn decode_latents(
    vae_decoder: &VaeDecoder,
    packed_latents: &Tensor,
    height: usize,
    width: usize,
    cfg: &PipelineConfig,
) -> Result<Tensor> {
    let vae_scale_factor = cfg.vae_cfg.spatial_compression_ratio();

    let latents = unpack_latents(packed_latents, height, width, vae_scale_factor)?;
    eprintln!("  After unpack: shape={:?}", latents.shape());
    let latents = normalize_latents(&latents, &cfg.vae_cfg.latents_mean, &cfg.vae_cfg.latents_std)?;
    eprintln!("  After normalize: shape={:?}", latents.shape());
    let latents = latents.squeeze(2)?;  // remove temporal dim [B, C, 1, H, W] -> [B, C, H, W]
    eprintln!("  After squeeze: shape={:?}", latents.shape());
    let image = vae_decoder.decode(&latents)?;

    Ok(image)
}