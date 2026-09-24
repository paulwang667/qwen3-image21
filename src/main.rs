use anyhow::{Error as E, Result};
use candle_core::{Device, DType};
use clap::Parser;

use qwen3_image21::pipeline::{denoise, decode_latents, PipelineConfig};
use qwen3_image21::service::{self, EngineConfig, GenRequest, ConditionImageInput, ModelPaths};
use qwen3_image21::transformer::set_flash_attention;
use qwen3_image21::vae::Config as VaeConfig;
use qwen3_image21::Precision;

#[derive(Parser)]
#[command(author, version, about = "Qwen-Image-2.1 Inference Engine")]
struct Args {
    /// Prompt for image generation (not needed in --serve mode)
    #[arg(long)]
    prompt: Option<String>,

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

    /// Keep the prefix KV cache in host memory, copying each layer's entries to
    /// the GPU only while that layer runs: saves ~2.2 GB of GPU memory per
    /// 1024² condition image, at the cost of PCIe transfers every step.
    #[arg(long)]
    kv_cache_cpu: bool,

    /// Weight/compute precision of the full-precision transformer and of the
    /// text encoder on the GPU. GGUF transformers always compute in F32 and the
    /// VAE always runs in F32.
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    precision: Precision,

    /// Run the text encoder (and vision tower) on the CPU in F32 instead of the
    /// GPU. On a GPU its layers are already streamed (~4 GB BF16 / ~6 GB F32
    /// peak), so this only helps when even that does not fit; ~15x slower.
    #[arg(long)]
    text_encoder_cpu: bool,

    /// Allow TF32 tensor cores for F32 matrix multiplications on CUDA (as
    /// PyTorch's `allow_tf32`): faster, ~1e-3 relative error per matmul. Only
    /// F32 matmuls are affected — GGUF linear layers use candle's quantized
    /// kernels, BF16 matmuls already use tensor cores.
    #[arg(long)]
    tf32: bool,

    /// Use FlashAttention v2 for unmasked attention (all steps after the first
    /// with the KV cache). Needs a build with --features flash-attn; F32
    /// inputs are cast to F16 for the kernel.
    #[arg(long)]
    flash_attn: bool,

    /// Run benchmark mode (multiple iterations)
    #[arg(long)]
    benchmark: bool,

    /// Number of benchmark iterations
    #[arg(long, default_value_t = 3)]
    benchmark_iterations: usize,

    /// Directory the model paths below default to: a `.gguf` transformer in it
    /// and the official repo (`text_encoder/`, `processor/`, `vae/`, optionally
    /// `transformer/`), either the directory itself or one subdirectory.
    #[arg(long, default_value = "models")]
    model_dir: String,

    /// Transformer: a GGUF file or the first shard of the official
    /// safetensors (default: the only .gguf in --model-dir, else the repo's
    /// transformer/)
    #[arg(long)]
    model_path: Option<String>,

    /// VAE safetensors (default: the repo's vae/ in --model-dir)
    #[arg(long)]
    vae_path: Option<String>,

    /// Text encoder directory: the official repo's `text_encoder/` (tokenizer
    /// read from the sibling `processor/`), or a dense Qwen3 stand-in — see
    /// text_encoder.rs (default: the repo's text_encoder/ in --model-dir;
    /// without one, random embeddings)
    #[arg(long)]
    text_encoder_path: Option<String>,

    /// Seed for the initial noise (default: random, printed so the run can be
    /// reproduced). Don't edit an image with the seed it was generated from:
    /// the identical starting noise locks the edit onto the original.
    #[arg(long)]
    seed: Option<u64>,

    // ── Server mode ────────────────────────────────────────────────────

    /// Start HTTP server (OpenAI-compatible image API) instead of a single
    /// generation. Build with --features server.
    #[arg(long)]
    serve: bool,

    /// HTTP server port (only with --serve)
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Comma-separated GPU device IDs for multi-GPU parallelism (only with
    /// --serve). Default: 0 (single GPU).
    #[arg(long)]
    device_ids: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // ── Server mode ────────────────────────────────────────────────────
    if args.serve {
        return run_server(&args);
    }

    // ── CLI mode ───────────────────────────────────────────────────────
    let prompt = args.prompt.as_deref().ok_or_else(|| {
        E::msg("--prompt is required (or use --serve for server mode)")
    })?;

    let device = match Device::cuda_if_available(0)? {
        Device::Cpu => Device::metal_if_available(0)?,
        d => d,
    };
    let dtype = args.precision.as_dtype();
    if args.tf32 && !device.is_cuda() {
        eprintln!("  --tf32 has no effect on {:?} (CUDA-only)", device);
    }
    candle_core::cuda::set_gemm_reduced_precision_f32(args.tf32);
    set_flash_attention(args.flash_attn)?;
    if args.kv_cache_cpu && device.is_metal() {
        eprintln!("  --kv-cache-cpu on Metal: Metal shares unified memory, so the host↔device copy is pure overhead.");
    }
    qwen3_image21::transformer::set_kv_cache_offload(args.kv_cache_cpu);

    eprintln!("Device: {:?}", device);
    eprintln!("Prompt: {}", prompt);
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

    let paths = ModelPaths::resolve(
        &args.model_dir,
        args.model_path.as_deref(),
        args.vae_path.as_deref(),
        args.text_encoder_path.as_deref(),
    )?;
    eprintln!("Transformer: {}", paths.transformer);
    eprintln!("VAE: {}", paths.vae);
    match &paths.text_encoder {
        Some(p) => eprintln!("Text encoder: {p}"),
        None => eprintln!("Text encoder: none found under {} — random embeddings, the output will not follow the prompt", args.model_dir),
    }

    let engine = EngineConfig {
        precision: args.precision,
        flash_attn: args.flash_attn,
        kv_cache_cpu: args.kv_cache_cpu,
        text_encoder_cpu: args.text_encoder_cpu,
        tf32: args.tf32,
    };

    if args.benchmark {
        return run_benchmark(&args, &paths, &engine, &device, dtype);
    }

    // Normal CLI mode: delegate to service::generate.
    let gen_req = GenRequest {
        prompt: prompt.to_string(),
        negative_prompt: args.negative_prompt.clone(),
        true_cfg_scale: args.true_cfg_scale,
        width: args.width,
        height: args.height,
        steps: args.steps,
        seed: args.seed,
        images: args.images.iter().map(|s| ConditionImageInput::Path(s.clone())).collect(),
        output_resolution: args.output_resolution,
        no_kv_cache: args.no_kv_cache,
    };
    let result = service::generate(&gen_req, &paths, &engine, &device)?;
    std::fs::write(&args.output, &result.png)?;
    println!("Image saved to: {}", args.output);
    Ok(())
}

/// Start the HTTP server.
fn run_server(args: &Args) -> Result<()> {
    #[cfg(not(feature = "server"))]
    {
        let _ = &args; // suppress unused warning when server feature is off
        eprintln!("Server mode requires building with --features server");
        eprintln!("  Example: cargo build --release --features 'metal server'  (macOS)");
        eprintln!("  Example: cargo build --release --features 'cuda server'   (CUDA)");
        std::process::exit(1);
    }

    #[cfg(feature = "server")]
    {
        let device_ids = qwen3_image21::server::parse_device_ids(&args.device_ids);
        let paths = ModelPaths::resolve(
            &args.model_dir,
            args.model_path.as_deref(),
            args.vae_path.as_deref(),
            args.text_encoder_path.as_deref(),
        )?;
        let engine = EngineConfig {
            precision: args.precision,
            flash_attn: args.flash_attn,
            kv_cache_cpu: args.kv_cache_cpu,
            text_encoder_cpu: args.text_encoder_cpu,
            tf32: args.tf32,
        };
        // Set global settings once before creating workers.
        candle_core::cuda::set_gemm_reduced_precision_f32(args.tf32);
        set_flash_attention(args.flash_attn)?;
        qwen3_image21::transformer::set_kv_cache_offload(args.kv_cache_cpu);

        eprintln!("Transformer: {}", paths.transformer);
        eprintln!("VAE: {}", paths.vae);
        match &paths.text_encoder {
            Some(p) => eprintln!("Text encoder: {p}"),
            None => eprintln!("Text encoder: none — image-conditioned generation will not be available"),
        }

        let addr = format!("0.0.0.0:{}", args.port);
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(qwen3_image21::server::run(&addr, device_ids, paths, engine))?;
        Ok(())
    }
}

/// Benchmark mode: transformer + VAE loaded together for repeated runs.
fn run_benchmark(
    args: &Args,
    paths: &ModelPaths,
    engine: &EngineConfig,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    let transformer_dtype = if paths.transformer.ends_with(".gguf") {
        DType::F32
    } else {
        dtype
    };
    let pipeline_cfg = PipelineConfig {
        vae_cfg: VaeConfig::default(),
        device: device.clone(),
        dtype: transformer_dtype,
    };
    let (encoder_device, encoder_dtype) = if engine.text_encoder_cpu {
        (Device::Cpu, DType::F32)
    } else {
        (device.clone(), dtype)
    };

    let condition_images = args
        .images
        .iter()
        .map(|p| qwen3_image21::condition_image::load(p, args.output_resolution, device))
        .collect::<Result<Vec<_>>>()?;
    let (default_w, default_h) = condition_images.last().map_or((1024, 1024), |c| (c.width, c.height));
    let (width, height) = (args.width.unwrap_or(default_w), args.height.unwrap_or(default_h));
    let seed = args.seed.unwrap_or_else(rand::random);

    let do_cfg = args.true_cfg_scale > 1.0 && args.negative_prompt.is_some();
    let (prompt_emb, negative_emb) = service::encode_prompt(
        paths.text_encoder.as_deref(),
        &args.prompt.clone().unwrap_or_default(),
        args.negative_prompt.as_deref().filter(|_| do_cfg),
        &condition_images,
        device,
        &encoder_device,
        encoder_dtype,
    )?;
    let guidance = negative_emb.as_ref().map(|neg| (neg, args.true_cfg_scale));
    let condition = if condition_images.is_empty() {
        None
    } else {
        Some(service::encode_condition_images(&paths.vae, &condition_images, device)?)
    };

    eprintln!("[Phase 2] Loading transformer...");
    let transformer = service::load_transformer(&paths.transformer, device, transformer_dtype)?;

    eprintln!("[Phase 2/3] Loading VAE...");
    let vae_decoder = service::load_vae(&paths.vae, device, DType::F32)?;

    // Warmup
    eprintln!("Warming up...");
    let _ = denoise(
        &transformer,
        &prompt_emb,
        height,
        width,
        args.steps,
        guidance,
        condition.as_ref(),
        !args.no_kv_cache,
        seed,
        &pipeline_cfg,
    )?;

    // Timed iterations
    let mut times = Vec::with_capacity(args.benchmark_iterations);
    let mut last_image = None;
    for i in 1..=args.benchmark_iterations {
        eprintln!("Benchmark iteration {}/{}", i, args.benchmark_iterations);
        let start = std::time::Instant::now();
        let latents = denoise(
            &transformer,
            &prompt_emb,
            height,
            width,
            args.steps,
            guidance,
            condition.as_ref(),
            !args.no_kv_cache,
            seed,
            &pipeline_cfg,
        )?;
        let img = decode_latents(&vae_decoder, &latents, height, width, &pipeline_cfg)?;
        let elapsed = start.elapsed();
        times.push(elapsed);
        eprintln!("  Time: {:.2}s", elapsed.as_secs_f32());
        last_image = Some(img);
    }
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

    let image = last_image.ok_or_else(|| E::msg("Benchmark generation failed"))?;
    let png = service::image_to_png_bytes(&image, width, height)?;
    std::fs::write(&args.output, &png)?;
    println!("Image saved to: {}", args.output);
    Ok(())
}
