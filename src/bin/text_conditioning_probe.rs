//! Diagnostic: isolate whether text conditioning actually influences the
//! transformer's noise prediction, independent of the denoising loop's
//! accumulated drift. Fixed initial latents + fixed timestep, three forward
//! passes: prompt A, prompt B (different), and prompt A again (determinism
//! control). If conditioning works, noise_pred(A) should differ from
//! noise_pred(B) by roughly as much as real content differs, while
//! noise_pred(A) vs noise_pred(A-repeat) should be ~identical (no dropout at
//! inference, so any nonzero diff there is itself a bug signal).
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use qwen3_image21::pipeline::TransformerType;
use qwen3_image21::transformer::{Config as TransformerConfig, QwenImageTransformer};
use qwen3_image21::vae::Config as VaeConfig;

fn mean_abs(x: &Tensor) -> Result<f64> {
    Ok(x.abs()?.mean_all()?.to_scalar::<f32>()? as f64)
}

fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a = a.flatten_all()?;
    let b = b.flatten_all()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()? as f64;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
    Ok(dot / (na * nb))
}

fn main() -> Result<()> {
    let device = match Device::cuda_if_available(0)? {
        Device::Cpu => Device::metal_if_available(0)?,
        d => d,
    };
    println!("Using device: {device:?}");

    let text_encoder_path = std::env::args().nth(1)
        .unwrap_or_else(|| "models/Qwen-Image-2.1-official/text_encoder".to_string());
    let transformer_path = std::env::args().nth(2)
        .unwrap_or_else(|| "models/Qwen-Image-2.1-official/transformer/diffusion_pytorch_model-00001-of-00002.safetensors".to_string());
    let prompt_a = std::env::args().nth(3).unwrap_or_else(|| "a red apple on a wooden table".to_string());
    let prompt_b = std::env::args().nth(4).unwrap_or_else(|| "a blue spaceship flying through outer space".to_string());
    let height: usize = 256;
    let width: usize = 256;

    // ── Text encoding ───────────────────────────────────────────────
    println!("Encoding prompt A: {prompt_a:?}");
    let mut encoder_a = qwen3_image21::text_encoder::TextEncoder::load(&text_encoder_path, 4096, device.clone())?;
    let emb_a = encoder_a.encode(&prompt_a)?;
    drop(encoder_a);
    println!("Encoding prompt B: {prompt_b:?}");
    let mut encoder_b = qwen3_image21::text_encoder::TextEncoder::load(&text_encoder_path, 4096, device.clone())?;
    let emb_b = encoder_b.encode(&prompt_b)?;
    drop(encoder_b);
    println!("emb_a.shape={:?} emb_b.shape={:?}", emb_a.shape(), emb_b.shape());

    // Pad the shorter embedding's sequence to match the longer one (repeat the
    // real content, not zero-padding, to avoid injecting a spurious "different
    // sequence length" confound) so both forward passes use identical img/txt
    // token counts and RoPE position layouts.
    let seq_a = emb_a.dim(1)?;
    let seq_b = emb_b.dim(1)?;
    let seq_len = seq_a.max(seq_b);
    let pad_to = |emb: &Tensor, seq: usize| -> Result<Tensor> {
        if seq == seq_len {
            return Ok(emb.clone());
        }
        let last = emb.narrow(1, seq - 1, 1)?;
        let pad = last.expand((1, seq_len - seq, emb.dim(2)?))?.contiguous()?;
        Ok(Tensor::cat(&[emb, &pad], 1)?)
    };
    let emb_a = pad_to(&emb_a, seq_a)?;
    let emb_b = pad_to(&emb_b, seq_b)?;

    // ── Load transformer (official non-quantized safetensors) ─────────
    println!("Loading transformer...");
    let paths = qwen3_image21::safetensors_util::resolve_safetensors_paths(&transformer_path)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &device)? };
    let cfg = TransformerConfig::default();
    let transformer = TransformerType::NonQuantized(QwenImageTransformer::new(&cfg, vb)?);

    // ── Fixed inputs shared across all three forward passes ───────────
    let vae_cfg = VaeConfig::default();
    let scale = vae_cfg.spatial_compression_ratio();
    let latent_h = height / scale;
    let latent_w = width / scale;
    let seq_img = latent_h * latent_w;

    let latents = Tensor::randn(0.0f32, 1.0f32, (1, seq_img, vae_cfg.z_dim), &device)?;
    let timestep = Tensor::new(&[500.0f32], &device)?;
    let timestep_emb = qwen3_image21::scheduler::timestep_embedding(&timestep, 256)?;
    let img_ids = qwen3_image21::rope::compute_img_ids(1, latent_h, latent_w, &device)?;
    let txt_ids = qwen3_image21::rope::compute_txt_ids(1, seq_len, 0, &device)?;

    println!("Running forward pass: prompt A...");
    let pred_a = transformer.forward(&latents, &timestep_emb, &emb_a, None, &img_ids, &txt_ids, height, width)?;
    println!("Running forward pass: prompt B...");
    let pred_b = transformer.forward(&latents, &timestep_emb, &emb_b, None, &img_ids, &txt_ids, height, width)?;
    println!("Running forward pass: prompt A again (determinism control)...");
    let pred_a2 = transformer.forward(&latents, &timestep_emb, &emb_a, None, &img_ids, &txt_ids, height, width)?;

    println!();
    println!("=== Results ===");
    println!("mean|pred_a|              = {:.6}", mean_abs(&pred_a)?);
    println!("mean|pred_a - pred_a2|    = {:.6}  (determinism control, expect ~0)", mean_abs(&(&pred_a - &pred_a2)?)?);
    println!("cosine(pred_a, pred_a2)   = {:.6}  (expect ~1.0)", cosine(&pred_a, &pred_a2)?);
    println!("mean|pred_a - pred_b|     = {:.6}  (different prompts)", mean_abs(&(&pred_a - &pred_b)?)?);
    println!("cosine(pred_a, pred_b)    = {:.6}  (expect noticeably < 1.0 if conditioning works)", cosine(&pred_a, &pred_b)?);
    println!("(pred_a/pred_b shape={:?} — the transformer already returns image-token predictions only)", pred_a.shape());

    Ok(())
}
