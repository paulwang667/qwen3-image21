use candle_core::{Result, Tensor, Device, DType};
use crate::joint_layout::ConditionTokens;
use crate::transformer::{QwenImageTransformer, TextKvCache};
use crate::quantized_transformer::QwenImageTransformerQuantized;
use crate::scheduler::FlowMatchEuler;
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

    /// General step: optional condition images and an optional prefix K/V cache
    /// (see `QwenImageTransformer::forward_conditioned`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_conditioned(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        text_emb: &Tensor,
        height: usize,
        width: usize,
        condition: Option<&ConditionTokens>,
        cache: Option<&mut Option<TextKvCache>>,
    ) -> Result<Tensor> {
        match self {
            Self::NonQuantized(t) => t.forward_conditioned(latents, Some(text_emb), None, timestep, height, width, condition, cache),
            Self::Quantized(t) => t.forward_conditioned(latents, Some(text_emb), None, timestep, height, width, condition, cache),
        }
    }

    /// Forward pass reusing the text prefix's K/V across steps (see
    /// `transformer::TextKvCache`); `cache` must belong to this `text_emb`.
    pub fn forward_with_text_cache(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        text_emb: &Tensor,
        text_emb_mask: Option<&Tensor>,
        height: usize,
        width: usize,
        cache: &mut Option<TextKvCache>,
    ) -> Result<Tensor> {
        match self {
            Self::NonQuantized(t) => t.forward_with_text_cache(latents, text_emb, text_emb_mask, timestep, height, width, cache),
            Self::Quantized(t) => t.forward_with_text_cache(latents, text_emb, text_emb_mask, timestep, height, width, cache),
        }
    }
}

/// An encoded prompt. `image_slots` marks its `<|image_pad|>` tokens when it
/// was encoded with condition images.
pub struct PromptEmbeds {
    pub embeds: Tensor,
    pub image_slots: Option<Vec<bool>>,
}

/// Packed VAE latents of the condition images, shared by the positive and
/// negative prompt.
pub struct ConditionLatents {
    /// `[1, sum(h*w), 64]`, images concatenated in prompt order.
    pub latents: Tensor,
    /// Latent `(h, w)` of each image.
    pub shapes: Vec<(usize, usize)>,
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
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &TransformerType,
    prompt: &PromptEmbeds,
    height: usize,
    width: usize,
    num_inference_steps: usize,
    guidance: Option<(&PromptEmbeds, f32)>,
    condition: Option<&ConditionLatents>,
    use_kv_cache: bool,
    seed: u64,
    cfg: &PipelineConfig,
) -> Result<Tensor> {
    let prompt_emb = &prompt.embeds;
    let batch_size = prompt_emb.dim(0)?;
    let num_channels_latents = cfg.vae_cfg.z_dim;
    let vae_scale_factor = cfg.vae_cfg.spatial_compression_ratio();
    let latent_h = height / vae_scale_factor;
    let latent_w = width / vae_scale_factor;

    // Initial noise [B, C, T, H, W] (temporal at index 2), drawn on the host from
    // `seed` so a seed gives the same noise on every device: candle's device RNGs
    // start from a fixed state in every process (CUDA/Metal) or can't be seeded
    // at all (CPU). A fixed noise is not harmless — editing an image generated
    // from the same noise locks the trajectory onto it (oversaturated output).
    let shape = (batch_size, num_channels_latents, 1, latent_h, latent_w);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let noise: Vec<f32> = (0..batch_size * num_channels_latents * latent_h * latent_w)
        .map(|_| rand_distr::Distribution::<f32>::sample(&rand_distr::StandardNormal, &mut rng))
        .collect();
    let latents = Tensor::from_vec(noise, shape, &cfg.device)?;
    let mut packed_latents = pack_latents(&latents)?.to_dtype(cfg.dtype)?;
    let prompt_emb = prompt_emb.to_dtype(cfg.dtype)?;
    // Optional true CFG: `(negative_prompt_emb, true_cfg_scale)`.
    let guidance = match guidance {
        Some((neg, scale)) => Some((neg.embeds.to_dtype(cfg.dtype)?, neg.image_slots.as_deref(), scale)),
        None => None,
    };
    let cond_latents = condition.map(|c| c.latents.to_dtype(cfg.dtype)).transpose()?;

    // Resolution-dependent shifted sigma schedule (see scheduler::FlowMatchEuler docs) —
    // image_seq_len is the packed latent's token count (patch_size=1, so latent_h*latent_w).
    let scheduler = FlowMatchEuler::new(num_inference_steps, latent_h * latent_w);
    let timesteps = scheduler.timesteps();

    eprintln!("  Initial packed_latents shape: {:?}", packed_latents.shape());
    eprintln!("  prompt_emb shape: {:?}", prompt_emb.shape());

    // One prefix K/V cache per prompt (positive, and negative under CFG); the
    // prefix covers the text and any condition images.
    let mut cache: Option<TextKvCache> = None;
    let mut neg_cache: Option<TextKvCache> = None;
    let predict = |text: &Tensor, slots: Option<&[bool]>, timestep: &Tensor, latents: &Tensor, cache: &mut Option<TextKvCache>| -> Result<Tensor> {
        let condition = match (condition, &cond_latents, slots) {
            (Some(c), Some(latents), Some(text_image_mask)) => Some(ConditionTokens { latents, shapes: &c.shapes, text_image_mask }),
            (Some(_), _, None) => candle_core::bail!("condition images need prompts encoded with their image slots"),
            _ => None,
        };
        let cache = if use_kv_cache { Some(cache) } else { None };
        transformer.forward_conditioned(latents, timestep, text, height, width, condition.as_ref(), cache)
    };

    // Denoising loop
    for i in 0..num_inference_steps {
        let t = timesteps[i];
        // [B] (batch=1); `timestep_embedding` does its own unsqueeze to broadcast
        // against the exponent table, so this must stay 1-D here. f32, not f64:
        // Metal has no F64->F32 dtype-cast kernel.
        let timestep = Tensor::new(&[t], &cfg.device)?
            .to_dtype(packed_latents.dtype())?;

        // Apply timestep embedding (sinusoidal -> 256 dim)
        let timestep = crate::scheduler::timestep_embedding(&timestep, 256)?
            .to_dtype(packed_latents.dtype())?;

        let noise_pred = predict(&prompt_emb, prompt.image_slots.as_deref(), &timestep, &packed_latents, &mut cache)?;
        // Upstream QwenImage21Pipeline: neg + scale * (cond - neg), no rescaling.
        let noise_pred = match &guidance {
            Some((neg_emb, neg_slots, scale)) => {
                let neg_pred = predict(neg_emb, *neg_slots, &timestep, &packed_latents, &mut neg_cache)?;
                (&neg_pred + ((&noise_pred - &neg_pred)? * *scale as f64)?)?
            }
            None => noise_pred,
        };

        if i == 0 {
            eprintln!("  noise_pred shape: {:?}", noise_pred.shape());
        }

        packed_latents = scheduler.step(&noise_pred, &packed_latents, i)?;

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