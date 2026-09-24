use anyhow::{Error as E, Result};
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;

use qwen3_image21::pipeline::{ConditionLatents, PipelineConfig, PromptEmbeds, TransformerType, denoise, decode_latents};
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

    /// Image height (default: 1024, or the condition image's aspect at
    /// --output-resolution when --image is given)
    #[arg(long)]
    height: Option<usize>,

    /// Image width (see --height)
    #[arg(long)]
    width: Option<usize>,

    /// Condition image for image-conditioned generation (editing / reference);
    /// repeat for several. Read by the Qwen3-VL vision tower and VAE-encoded
    /// into the transformer's sequence, as upstream QwenImage21Pipeline.
    #[arg(long = "image")]
    images: Vec<String>,

    /// Side length of the square area condition images are resized to (at their
    /// own aspect ratio, multiples of 32); also sets the default output size.
    #[arg(long, default_value_t = 1024)]
    output_resolution: usize,

    /// Number of inference steps (upstream QwenImage21Pipeline default: 40)
    #[arg(long, default_value_t = 40)]
    steps: usize,

    /// Negative prompt for true classifier-free guidance (only used when
    /// --true-cfg-scale > 1, as upstream)
    #[arg(long)]
    negative_prompt: Option<String>,

    /// True CFG scale. Qwen-Image-2.1 is meant to be sampled without guidance,
    /// hence the upstream default of 1.0 (off).
    #[arg(long, default_value_t = 1.0)]
    true_cfg_scale: f32,

    /// Recompute the text prefix every step instead of caching its per-layer
    /// K/V (the cache is exact under causal_condition; this is for A/B checks)
    #[arg(long)]
    no_kv_cache: bool,

    /// Use quantized model
    #[arg(long)]
    quantized: bool,

    /// Weight/compute precision of the full-precision transformer and of the
    /// text encoder on the GPU. GGUF transformers always compute in F32 and the
    /// VAE always runs in F32.
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    precision: Precision,

    /// Run the text encoder (and vision tower) on the CPU in F32 instead of the
    /// GPU: removes its ~34 GB F32 / ~17 GB BF16 from GPU memory at the cost of
    /// slower prompt encoding.
    #[arg(long)]
    text_encoder_cpu: bool,

    /// Allow TF32 tensor cores for F32 matrix multiplications on CUDA (as
    /// PyTorch's `allow_tf32`): faster, ~1e-3 relative error per matmul. Only
    /// F32 matmuls are affected — GGUF linear layers use candle's quantized
    /// kernels, BF16 matmuls already use tensor cores.
    #[arg(long)]
    tf32: bool,

    /// Use FlashAttention v2 for unmasked attention (all steps after the first
    /// with the KV cache). Needs a build with `--features flash-attn`; F32
    /// inputs are cast to F16 for the kernel.
    #[arg(long)]
    flash_attn: bool,

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

    /// Text encoder directory: the official repo's `text_encoder/` (tokenizer
    /// read from the sibling `processor/`), or a dense Qwen3 stand-in — see
    /// text_encoder.rs. Omit for random embeddings.
    #[arg(long)]
    text_encoder_path: Option<String>,

    /// Seed for the initial noise (default: random, printed so the run can be
    /// reproduced). Don't edit an image with the seed it was generated from:
    /// the identical starting noise locks the edit onto the original.
    #[arg(long)]
    seed: Option<u64>,
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
        // ComfyUI-style conversions (e.g. unsloth's) prefix every tensor name.
        let vb = mapped_vb.inner().clone();
        let vb = if vb.contains_key("model.diffusion_model.img_in.weight") { vb.pp("model.diffusion_model") } else { vb };
        let transformer = QwenImageTransformerQuantized::new(&cfg, vb)?;
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
/// Loads the text encoder, encodes the prompt (with any condition images), then
/// drops the model. The returned tensors are independent of the model.
fn encode_prompt(
    text_encoder_path: Option<&str>,
    prompt: &str,
    negative_prompt: Option<&str>,
    images: &[qwen3_image21::condition_image::ConditionImage],
    device: &Device,
    encoder_device: &Device,
    encoder_dtype: DType,
) -> Result<(PromptEmbeds, Option<PromptEmbeds>)> {
    eprintln!("[Phase 1] Text encoding");
    let start = std::time::Instant::now();
    let embs = match text_encoder_path {
        Some(path) => {
            eprintln!("  Loading text encoder from: {} ({:?} on {:?})", path, encoder_dtype, encoder_device);
            // Dropped when this scope ends.
            let mut encoder = qwen3_image21::text_encoder::TextEncoder::load(path, 4096, encoder_device.clone(), encoder_dtype)?;
            let mut encode = |p: &str| -> Result<PromptEmbeds> {
                let (embeds, image_slots) = if images.is_empty() {
                    (encoder.encode(p)?, None)
                } else {
                    // Upstream encodes the negative prompt with the same condition images.
                    let (embeds, slots) = encoder.encode_with_images(p, images)?;
                    (embeds, Some(slots))
                };
                Ok(PromptEmbeds { embeds: embeds.to_device(device)?, image_slots })
            };
            let negative = negative_prompt.map(&mut encode).transpose()?;
            (encode(prompt)?, negative)
        }
        None => {
            if !images.is_empty() {
                return Err(E::msg("--image needs --text-encoder-path (the vision tower is part of the text encoder)"));
            }
            eprintln!("  No text encoder path, using random embeddings");
            let random = || -> Result<PromptEmbeds> {
                // f32: Metal has no F64 rand_uniform
                Ok(PromptEmbeds { embeds: Tensor::randn(0.0f32, 1.0f32, (1, 256, 4096), device)?, image_slots: None })
            };
            (random()?, negative_prompt.map(|_| random()).transpose()?)
        }
    };
    release_freed_memory(device)?;
    eprintln!("  Text encoding done in {:.2}s. Encoder released.", start.elapsed().as_secs_f32());
    Ok(embs)
}

/// VAE-encodes the condition images into packed latents for the transformer.
fn encode_condition_images(
    vae_path: &str,
    images: &[qwen3_image21::condition_image::ConditionImage],
    device: &Device,
) -> Result<ConditionLatents> {
    eprintln!("[Phase 1b] Encoding {} condition image(s) with the VAE", images.len());
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path.to_string()], DType::F32, device)? };
    let encoder = qwen3_image21::vae::VaeEncoder::new(&VaeConfig::default(), vb)?;
    let mut packed = Vec::with_capacity(images.len());
    let mut shapes = Vec::with_capacity(images.len());
    for img in images {
        let latents = encoder.encode(&img.vae_input)?; // [1, 64, h, w]
        let (_, _, h, w) = latents.dims4()?;
        packed.push(qwen3_image21::vae::pack_latents(&latents.unsqueeze(2)?)?);
        shapes.push((h, w));
    }
    let latents = Tensor::cat(&packed, 1)?;
    drop(encoder);
    release_freed_memory(device)?;
    Ok(ConditionLatents { latents, shapes })
}

/// candle's CUDA backend allocates from stream-ordered memory pools, which keep
/// freed memory reserved until the next device synchronize. Without this, a
/// dropped model's memory stays reserved, and the next model's large buffers
/// don't fit into its fragmented blocks, so the phases' peaks add up (measured:
/// the transformer's 17 GB stayed reserved under the VAE decode's 10 GB).
fn release_freed_memory(device: &Device) -> Result<()> {
    Ok(device.synchronize()?)
}

/// Save image tensor [-1,1] to PNG file. The VAE decoder outputs 4 channels
/// (RGBA, per the real checkpoint's `conv_out` weight shape `[4, 144, 3, 3]`),
/// so the alpha channel must be dropped before handing bytes to an Rgb image
/// buffer — `ImageBuffer::from_vec` only checks the buffer is at least
/// width*height*3 bytes, not exactly, so an un-dropped 4th channel is silently
/// accepted and misread at the wrong stride, producing a periodic stripe artifact.
fn save_image(image: &Tensor, width: usize, height: usize, output: &str) -> Result<()> {
    let image = image.clamp(-1.0, 1.0)?;
    let image = image.affine(0.5, 0.5)?;   // (x + 1) * 0.5
    let image = image.affine(255.0, 0.0)?; // * 255
    let image = image.to_dtype(candle_core::DType::U8)?;
    let image = image.i(0)?;               // [C, H, W]
    let image = image.permute((1, 2, 0))?; // [H, W, C]

    let channels = image.dim(2)?;
    let pixels = image.flatten_all()?.to_vec1::<u8>().unwrap();
    let rgb: Vec<u8> = pixels.chunks(channels).flat_map(|px| [px[0], px[1], px[2]]).collect();

    let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_vec(width as u32, height as u32, rgb).unwrap();
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
    candle_core::cuda::set_gemm_reduced_precision_f32(args.tf32);
    qwen3_image21::transformer::set_flash_attention(args.flash_attn)?;

    eprintln!("Device: {:?}", device);
    eprintln!("Prompt: {}", args.prompt);
    // Condition images are resized once, up front; with them the output size
    // defaults to the last image's size (upstream derives both from the same
    // `calculate_dimensions(output_resolution², aspect)`).
    let condition_images = args
        .images
        .iter()
        .map(|p| qwen3_image21::condition_image::load(p, args.output_resolution, &device))
        .collect::<Result<Vec<_>>>()?;
    let (default_w, default_h) = condition_images.last().map_or((1024, 1024), |c| (c.width, c.height));
    let (width, height) = (args.width.unwrap_or(default_w), args.height.unwrap_or(default_h));
    eprintln!("Size: {}x{}", width, height);
    for (path, c) in args.images.iter().zip(&condition_images) {
        eprintln!("Condition image: {path} -> {}x{}", c.width, c.height);
    }
    eprintln!("Steps: {}", args.steps);
    let seed = args.seed.unwrap_or_else(rand::random);
    eprintln!("Seed: {seed}");
    eprintln!("Precision: {:?}", args.precision);
    eprintln!("Quantized: {}", args.quantized);

    if args.model_path.is_none() {
        return Err(E::msg("Model path (--model-path) is required"));
    }

    let model_path = args.model_path.as_ref().unwrap();
    let vae_path = args.vae_path.as_ref().map(|s| s.as_str()).unwrap_or(model_path.as_str());

    // candle's quantized matmul takes F32 input, so GGUF transformers compute in F32.
    let transformer_dtype = if model_path.ends_with(".gguf") { DType::F32 } else { dtype };
    let pipeline_cfg = PipelineConfig {
        vae_cfg: VaeConfig::default(),
        device: device.clone(),
        dtype: transformer_dtype,
    };
    let (encoder_device, encoder_dtype) = if args.text_encoder_cpu { (Device::Cpu, DType::F32) } else { (device.clone(), dtype) };

    // ── Phase 1: Text encoding (load → encode → drop) ──────────────
    // Upstream enables true CFG only with scale > 1 AND a negative prompt.
    let do_cfg = args.true_cfg_scale > 1.0 && args.negative_prompt.is_some();
    if args.true_cfg_scale > 1.0 && args.negative_prompt.is_none() {
        eprintln!("  --true-cfg-scale > 1 but no --negative-prompt: guidance disabled");
    } else if args.true_cfg_scale <= 1.0 && args.negative_prompt.is_some() {
        eprintln!("  --negative-prompt ignored: guidance needs --true-cfg-scale > 1");
    }
    let (prompt_emb, negative_emb) = encode_prompt(
        args.text_encoder_path.as_deref(),
        &args.prompt,
        args.negative_prompt.as_deref().filter(|_| do_cfg),
        &condition_images,
        &device,
        &encoder_device,
        encoder_dtype,
    )?;
    let guidance = negative_emb.as_ref().map(|neg| (neg, args.true_cfg_scale));
    let condition = if condition_images.is_empty() {
        None
    } else {
        Some(encode_condition_images(vae_path, &condition_images, &device)?)
    };
    // Text encoder is dropped here; ~9.5 GB freed.

    // ── Phase 2+3: Diffusion ───────────────────────────────────────
    let image = if args.benchmark {
        // Benchmark: transformer + VAE loaded together for repeated runs.
        eprintln!("[Phase 2] Loading transformer...");
        let transformer = load_transformer(model_path, &device, args.quantized, transformer_dtype)?;

        eprintln!("[Phase 2/3] Loading VAE...");
        let vae_decoder = load_vae(vae_path, &device, DType::F32)?;

        // Warmup
        eprintln!("Warming up...");
        let _ = denoise(&transformer, &prompt_emb, height, width, args.steps, guidance, condition.as_ref(), !args.no_kv_cache, seed, &pipeline_cfg)?;

        // Timed iterations
        let mut times = Vec::with_capacity(args.benchmark_iterations);
        let mut last_image = None;
        for i in 1..=args.benchmark_iterations {
            eprintln!("Benchmark iteration {}/{}", i, args.benchmark_iterations);
            let start = std::time::Instant::now();
            let latents = denoise(&transformer, &prompt_emb, height, width, args.steps, guidance, condition.as_ref(), !args.no_kv_cache, seed, &pipeline_cfg)?;
            let img = decode_latents(&vae_decoder, &latents, height, width, &pipeline_cfg)?;
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
        let transformer = load_transformer(model_path, &device, args.quantized, transformer_dtype)?;

        eprintln!("  Denoising ({} steps)...", args.steps);
        let start = std::time::Instant::now();
        let latents = denoise(&transformer, &prompt_emb, height, width, args.steps, guidance, condition.as_ref(), !args.no_kv_cache, seed, &pipeline_cfg)?;
        eprintln!("  Denoising done in {:.2}s", start.elapsed().as_secs_f32());

        // Debug hook: dump the real packed latents for offline VAE diagnostics
        // (e.g. checking whether the dup_up3d checkerboard artifact appears on
        // a genuine denoised latent, not just synthetic test tensors).
        if let Ok(path) = std::env::var("QWEN_DUMP_LATENTS_PATH") {
            let tensors = std::collections::HashMap::from([("packed_latents".to_string(), latents.clone())]);
            candle_core::safetensors::save(&tensors, &path)?;
            eprintln!("  Dumped packed_latents to {path}");
        }

        // Drop transformer before loading VAE — frees ~3.8-6.8 GB.
        drop(transformer);
        release_freed_memory(&device)?;
        eprintln!("  Transformer released.");

        eprintln!("[Phase 3] Loading VAE...");
        let vae_decoder = load_vae(vae_path, &device, DType::F32)?;

        eprintln!("  Decoding...");
        let decode_start = std::time::Instant::now();
        let image = decode_latents(&vae_decoder, &latents, height, width, &pipeline_cfg)?;
        eprintln!("  Decoding done in {:.2}s", decode_start.elapsed().as_secs_f32());

        drop(vae_decoder);
        eprintln!("  VAE released.");

        image
    };

    save_image(&image, width, height, &args.output)?;
    Ok(())
}
