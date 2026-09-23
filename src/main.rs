use anyhow::{Error as E, Result};
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;

use qwen3_image21::pipeline::{PipelineConfig, TransformerType, denoise, decode_latents};
use qwen3_image21::transformer::{Config as TransformerConfig, QwenImageTransformer};
use qwen3_image21::vae::{Config as VaeConfig, VaeDecoder};
use qwen3_image21::gguf_mapped::load_gguf_mapped;
use qwen3_image21::quantized_transformer::QwenImageTransformerQuantized;
use qwen3_image21::Precision;

#[derive(Parser)]
#[command(author, version, about = "Qwen-Image-2.1 Quantized Inference Engine")]
struct Args {
    /// Prompt for image generation
    #[arg(long)]
    prompt: String,

    /// Output image file
    #[arg(long, default_value = "output.png")]
    output: String,

    /// Image height
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// Image width
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Number of inference steps
    #[arg(long, default_value_t = 50)]
    steps: usize,

    /// Use quantized model
    #[arg(long)]
    quantized: bool,

    /// Compute precision: f32, f16, or bf16
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    precision: Precision,

    /// Run benchmark mode (multiple iterations)
    #[arg(long)]
    benchmark: bool,

    /// Number of benchmark iterations
    #[arg(long, default_value_t = 3)]
    benchmark_iterations: usize,

    /// Model path (GGUF or safetensors)
    #[arg(long)]
    model_path: Option<String>,

    /// VAE model path
    #[arg(long)]
    vae_path: Option<String>,

    /// Text encoder model directory (must contain config.json, model.safetensors,
    /// and tokenizer.json for a standard dense Qwen3 model — see text_encoder.rs)
    #[arg(long)]
    text_encoder_path: Option<String>,

    /// Seed for random generation
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn load_transformer(
    model_path: &str,
    device: &Device,
    quantized: bool,
    dtype: DType,
) -> Result<TransformerType> {
    if model_path.ends_with(".gguf") {
        eprintln!("  Loading transformer from GGUF (quantized)...");
        let mapped_vb = load_gguf_mapped(model_path, device)?;
        let cfg = TransformerConfig::default();
        let transformer = QwenImageTransformerQuantized::new(&cfg, mapped_vb.inner().clone())?;
        return Ok(TransformerType::Quantized(transformer));
    }
    // safetensors path (non-quantized) — possibly sharded (see resolve_safetensors_paths)
    let paths = qwen3_image21::safetensors_util::resolve_safetensors_paths(model_path)?;
    eprintln!("  Loading transformer from {} safetensors shard(s) (dtype={:?})...", paths.len(), dtype);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? };
    let cfg = TransformerConfig::default();
    let transformer = QwenImageTransformer::new(&cfg, vb)?;
    Ok(TransformerType::NonQuantized(transformer))
}

fn load_vae(vae_path: &str, device: &Device, dtype: DType) -> Result<VaeDecoder> {
    let cfg = VaeConfig::default();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path.to_string()], dtype, device)? };
    Ok(VaeDecoder::new(&cfg, vb)?)
}


/// Phase 1: Text encoding.
/// Loads the text encoder, encodes the prompt, then drops the model.
/// The returned Tensor is independent (lives on device, no model reference).
fn encode_prompt(
    text_encoder_path: Option<&str>,
    prompt: &str,
    device: &Device,
) -> Result<Tensor> {
    eprintln!("[Phase 1] Text encoding");
    let prompt_emb = match text_encoder_path {
        Some(path) => {
            eprintln!("  Loading text encoder from: {}", path);
            // Not the real Qwen3-Next text encoder (unimplemented — see
            // text_encoder.rs); a standard dense Qwen3 stand-in, tiled up to
            // joint_attention_dim. Dropped when this scope ends.
            let mut encoder = qwen3_image21::text_encoder::TextEncoder::load(path, 4096, device.clone())?;
            encoder.encode(prompt)?
        }
        None => {
            eprintln!("  No text encoder path, using random embeddings");
            Tensor::randn(0.0f32, 1.0f32, (1, 256, 4096), device)? // f32: Metal has no F64 rand_uniform
        }
    };
    eprintln!("  Text encoding done. Encoder released.");
    Ok(prompt_emb)
}

/// Save image tensor [-1,1] to PNG file.
fn save_image(image: &Tensor, width: usize, height: usize, output: &str) -> Result<()> {
    let image = image.clamp(-1.0, 1.0)?;
    let image = image.affine(0.5, 0.5)?;   // (x + 1) * 0.5
    let image = image.affine(255.0, 0.0)?; // * 255
    let image = image.to_dtype(candle_core::DType::U8)?;
    let image = image.i(0)?;               // [C, H, W]
    let image = image.permute((1, 2, 0))?; // [H, W, C]

    let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_vec(
            width as u32,
            height as u32,
            image.flatten_all()?.to_vec1::<u8>().unwrap(),
        )
        .unwrap();
    img.save(output)?;
    println!("Image saved to: {}", output);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Cargo.toml's "metal"/"cuda" features are opt-in per build (`cargo build
    // --features metal` on macOS, `--features cuda` on Linux/NVIDIA); whichever
    // wasn't compiled in has its is_available() check compile down to `false`,
    // so trying both here is safe regardless of which one this binary has.
    let device = match Device::cuda_if_available(0)? {
        Device::Cpu => Device::metal_if_available(0)?,
        d => d,
    };
    let dtype = args.precision.as_dtype();

    eprintln!("Device: {:?}", device);
    eprintln!("Prompt: {}", args.prompt);
    eprintln!("Size: {}x{}", args.width, args.height);
    eprintln!("Steps: {}", args.steps);
    eprintln!("Precision: {:?}", args.precision);
    eprintln!("Quantized: {}", args.quantized);

    if args.model_path.is_none() {
        return Err(E::msg("Model path (--model-path) is required"));
    }

    let model_path = args.model_path.as_ref().unwrap();
    let vae_path = args.vae_path.as_ref().map(|s| s.as_str()).unwrap_or(model_path.as_str());

    let pipeline_cfg = PipelineConfig {
        vae_cfg: VaeConfig::default(),
        device: device.clone(),
        dtype,
    };

    // ── Phase 1: Text encoding (load → encode → drop) ──────────────
    let prompt_emb = encode_prompt(
        args.text_encoder_path.as_deref(),
        &args.prompt,
        &device,
    )?;
    // Text encoder is dropped here; ~9.5 GB freed.

    // ── Phase 2+3: Diffusion ───────────────────────────────────────
    let image = if args.benchmark {
        // Benchmark: transformer + VAE loaded together for repeated runs.
        eprintln!("[Phase 2] Loading transformer...");
        let transformer = load_transformer(model_path, &device, args.quantized, dtype)?;

        eprintln!("[Phase 2/3] Loading VAE...");
        let vae_decoder = load_vae(vae_path, &device, dtype)?;

        // Warmup
        eprintln!("Warming up...");
        let _ = denoise(&transformer, &prompt_emb, args.height, args.width, args.steps, &pipeline_cfg)?;

        // Timed iterations
        let mut times = Vec::with_capacity(args.benchmark_iterations);
        let mut last_image = None;
        for i in 1..=args.benchmark_iterations {
            eprintln!("Benchmark iteration {}/{}", i, args.benchmark_iterations);
            let start = std::time::Instant::now();
            let latents = denoise(&transformer, &prompt_emb, args.height, args.width, args.steps, &pipeline_cfg)?;
            let img = decode_latents(&vae_decoder, &latents, args.height, args.width, &pipeline_cfg)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
            eprintln!("  Time: {:.2}s", elapsed.as_secs_f32());
            last_image = Some(img);
        }
        // Drop both models
        drop(transformer);
        drop(vae_decoder);
        eprintln!("Models released.");

        // Stats
        let mean = times.iter().map(|t| t.as_secs_f32()).sum::<f32>() / times.len() as f32;
        let min = times.iter().map(|t| t.as_secs_f32()).fold(f32::MAX, f32::min);
        let max = times.iter().map(|t| t.as_secs_f32()).fold(f32::MIN, f32::max);
        let steps_per_sec = args.steps as f32 / mean;

        eprintln!("\n=== Benchmark Results ===");
        eprintln!("Mean: {:.2}s ({:.2} steps/s)", mean, steps_per_sec);
        eprintln!("Min:  {:.2}s", min);
        eprintln!("Max:  {:.2}s", max);

        last_image.ok_or_else(|| E::msg("Benchmark generation failed"))?
    } else {
        // Normal mode: load transformer → denoise → drop → load VAE → decode → drop
        eprintln!("[Phase 2] Loading transformer...");
        let transformer = load_transformer(model_path, &device, args.quantized, dtype)?;

        eprintln!("  Denoising ({} steps)...", args.steps);
        let start = std::time::Instant::now();
        let latents = denoise(&transformer, &prompt_emb, args.height, args.width, args.steps, &pipeline_cfg)?;
        eprintln!("  Denoising done in {:.2}s", start.elapsed().as_secs_f32());

        // Drop transformer before loading VAE — frees ~3.8-6.8 GB.
        drop(transformer);
        eprintln!("  Transformer released.");

        eprintln!("[Phase 3] Loading VAE...");
        let vae_decoder = load_vae(vae_path, &device, dtype)?;

        eprintln!("  Decoding...");
        let decode_start = std::time::Instant::now();
        let image = decode_latents(&vae_decoder, &latents, args.height, args.width, &pipeline_cfg)?;
        eprintln!("  Decoding done in {:.2}s", decode_start.elapsed().as_secs_f32());

        drop(vae_decoder);
        eprintln!("  VAE released.");

        image
    };

    save_image(&image, args.width, args.height, &args.output)?;
    Ok(())
}
