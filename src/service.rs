//! Core inference service: request types and the phased `generate` function.
//! Shared by the CLI (`main.rs`) and the HTTP server (`server.rs`).
//!
//! Models are loaded in phases (text encoder → VAE encoder → transformer → VAE
//! decoder), each dropped before loading the next to control peak memory — the
//! same strategy as the original CLI. On a per-request basis this adds only a
//! few seconds of load overhead vs. keeping models resident.

use anyhow::{Error as E, Result};
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::pipeline::{
    ConditionLatents, PipelineConfig, PromptEmbeds, TransformerType, denoise, decode_latents,
};
use crate::transformer::{Config as TransformerConfig, QwenImageTransformer};
use crate::vae::{Config as VaeConfig, VaeDecoder};
use crate::gguf_mapped::load_gguf_mapped;
use crate::quantized_transformer::QwenImageTransformerQuantized;
use crate::Precision;

/// Resolved model paths, shared across requests (resolved once at startup).
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub transformer: String,
    pub vae: String,
    pub text_encoder: Option<String>,
}

impl ModelPaths {
    /// Resolve from the same `--model-dir` / override args the CLI uses.
    pub fn resolve(
        model_dir: &str,
        model_path: Option<&str>,
        vae_path: Option<&str>,
        text_encoder_path: Option<&str>,
    ) -> Result<Self> {
        let paths = crate::model_dir::resolve(
            std::path::Path::new(model_dir),
            model_path,
            vae_path,
            text_encoder_path,
        )?;
        Ok(Self {
            transformer: paths.transformer.to_string_lossy().into_owned(),
            vae: paths.vae.to_string_lossy().into_owned(),
            text_encoder: paths.text_encoder.map(|p| p.to_string_lossy().into_owned()),
        })
    }
}

/// Engine-level configuration (set once at startup, shared across requests).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub precision: Precision,
    pub flash_attn: bool,
    pub kv_cache_cpu: bool,
    pub text_encoder_cpu: bool,
    pub tf32: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
            flash_attn: false,
            kv_cache_cpu: false,
            text_encoder_cpu: false,
            tf32: false,
        }
    }
}

/// A condition image for image-conditioned generation — either a file path
/// (CLI) or raw bytes (HTTP upload).
#[derive(Debug, Clone)]
pub enum ConditionImageInput {
    Path(String),
    Bytes(Vec<u8>),
}

/// Per-request generation parameters.
#[derive(Debug, Clone)]
pub struct GenRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub true_cfg_scale: f32,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: usize,
    pub seed: Option<u64>,
    pub images: Vec<ConditionImageInput>,
    pub output_resolution: usize,
    pub no_kv_cache: bool,
}

/// Generation result: PNG bytes + metadata.
pub struct GenResult {
    pub png: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
}

/// Run the full inference pipeline. The device must be exclusively used by
/// this call (single-threaded). Models are loaded in phases and dropped
/// between phases to control peak memory.
pub fn generate(
    req: &GenRequest,
    paths: &ModelPaths,
    engine: &EngineConfig,
    device: &Device,
) -> Result<GenResult> {
    let dtype = engine.precision.as_dtype();
    // candle's quantized matmul takes F32 input, so GGUF transformers compute in F32.
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

    // Load condition images (from paths or bytes).
    let condition_images = req
        .images
        .iter()
        .map(|input| match input {
            ConditionImageInput::Path(p) => {
                crate::condition_image::load(p, req.output_resolution, device)
            }
            ConditionImageInput::Bytes(b) => {
                crate::condition_image::load_from_bytes(b, req.output_resolution, device)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let (default_w, default_h) = condition_images
        .last()
        .map_or((1024, 1024), |c| (c.width, c.height));
    let (width, height) = (
        req.width.unwrap_or(default_w),
        req.height.unwrap_or(default_h),
    );
    let seed = req.seed.unwrap_or_else(rand::random);

    eprintln!("Device: {:?}", device);
    eprintln!("Prompt: {}", req.prompt);
    eprintln!("Size: {}x{}", width, height);
    eprintln!("Steps: {}", req.steps);
    eprintln!("Seed: {seed}");
    eprintln!("Precision: {:?}", engine.precision);

    // ── Phase 1: Text encoding (load → encode → drop) ──────────────
    let do_cfg = req.true_cfg_scale > 1.0 && req.negative_prompt.is_some();
    if req.true_cfg_scale > 1.0 && req.negative_prompt.is_none() {
        eprintln!("  cfg_scale > 1 but no negative_prompt: guidance disabled");
    } else if req.true_cfg_scale <= 1.0 && req.negative_prompt.is_some() {
        eprintln!("  negative_prompt ignored: guidance needs cfg_scale > 1");
    }
    let (prompt_emb, negative_emb) = encode_prompt(
        paths.text_encoder.as_deref(),
        &req.prompt,
        req.negative_prompt.as_deref().filter(|_| do_cfg),
        &condition_images,
        device,
        &encoder_device,
        encoder_dtype,
    )?;
    let guidance = negative_emb
        .as_ref()
        .map(|neg| (neg, req.true_cfg_scale));
    let condition = if condition_images.is_empty() {
        None
    } else {
        Some(encode_condition_images(&paths.vae, &condition_images, device)?)
    };

    // ── Phase 2: Transformer → denoise → drop ──────────────────────
    eprintln!("[Phase 2] Loading transformer...");
    let transformer = load_transformer(&paths.transformer, device, transformer_dtype)?;
    eprintln!("  Denoising ({} steps)...", req.steps);
    let start = std::time::Instant::now();
    let latents = denoise(
        &transformer,
        &prompt_emb,
        height,
        width,
        req.steps,
        guidance,
        condition.as_ref(),
        !req.no_kv_cache,
        seed,
        &pipeline_cfg,
    )?;
    eprintln!("  Denoising done in {:.2}s", start.elapsed().as_secs_f32());

    // Debug hook: dump real packed latents for offline VAE diagnostics.
    if let Ok(path) = std::env::var("QWEN_DUMP_LATENTS_PATH") {
        let tensors = std::collections::HashMap::from([(
            "packed_latents".to_string(),
            latents.clone(),
        )]);
        candle_core::safetensors::save(&tensors, &path)?;
        eprintln!("  Dumped packed_latents to {path}");
    }

    drop(transformer);
    release_freed_memory(device)?;
    eprintln!("  Transformer released.");

    // ── Phase 3: VAE decode → drop ─────────────────────────────────
    eprintln!("[Phase 3] Loading VAE...");
    let vae_decoder = load_vae(&paths.vae, device, DType::F32)?;
    eprintln!("  Decoding...");
    let decode_start = std::time::Instant::now();
    let image = decode_latents(&vae_decoder, &latents, height, width, &pipeline_cfg)?;
    eprintln!("  Decoding done in {:.2}s", decode_start.elapsed().as_secs_f32());
    drop(vae_decoder);
    release_freed_memory(device)?;
    eprintln!("  VAE released.");

    let png = image_to_png_bytes(&image, width, height)?;
    Ok(GenResult {
        png,
        width,
        height,
        seed,
    })
}

// ── Model loading helpers (moved from main.rs, now shared) ──────────────

pub fn load_transformer(
    model_path: &str,
    device: &Device,
    dtype: DType,
) -> Result<TransformerType> {
    if model_path.ends_with(".gguf") {
        eprintln!("  Loading transformer from GGUF (quantized)...");
        let mapped_vb = load_gguf_mapped(model_path, device)?;
        let cfg = TransformerConfig::default();
        // ComfyUI-style conversions (e.g. unsloth's) prefix every tensor name.
        let vb = mapped_vb.inner().clone();
        let vb = if vb.contains_key("model.diffusion_model.img_in.weight") {
            vb.pp("model.diffusion_model")
        } else {
            vb
        };
        let transformer = QwenImageTransformerQuantized::new(&cfg, vb)?;
        return Ok(TransformerType::Quantized(transformer));
    }
    // safetensors path (non-quantized) — possibly sharded.
    let paths = crate::safetensors_util::resolve_safetensors_paths(model_path)?;
    eprintln!(
        "  Loading transformer from {} safetensors shard(s) (dtype={:?})...",
        paths.len(),
        dtype
    );
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? };
    let cfg = TransformerConfig::default();
    let transformer = QwenImageTransformer::new(&cfg, vb)?;
    Ok(TransformerType::NonQuantized(transformer))
}

pub fn load_vae(vae_path: &str, device: &Device, dtype: DType) -> Result<VaeDecoder> {
    let cfg = VaeConfig::default();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path.to_string()], dtype, device)? };
    Ok(VaeDecoder::new(&cfg, vb)?)
}

/// Phase 1: Text encoding. Loads the text encoder, encodes the prompt (with
/// any condition images), then drops the model.
pub fn encode_prompt(
    text_encoder_path: Option<&str>,
    prompt: &str,
    negative_prompt: Option<&str>,
    images: &[crate::condition_image::ConditionImage],
    device: &Device,
    encoder_device: &Device,
    encoder_dtype: DType,
) -> Result<(PromptEmbeds, Option<PromptEmbeds>)> {
    eprintln!("[Phase 1] Text encoding");
    let start = std::time::Instant::now();
    let embs = match text_encoder_path {
        Some(path) => {
            eprintln!(
                "  Loading text encoder from: {} ({:?} on {:?})",
                path, encoder_dtype, encoder_device
            );
            // Dropped when this scope ends.
            // On a GPU the language-model layers are streamed in one at a time:
            // same speed (loading dominates either way) and bit-identical output,
            // at ~4 GB of GPU memory instead of ~17 GB (BF16) / ~34 GB (F32).
            let mut encoder = if !encoder_device.is_cpu() {
                crate::text_encoder::TextEncoder::load_streamed(
                    path,
                    4096,
                    encoder_device.clone(),
                    encoder_dtype,
                )?
            } else {
                crate::text_encoder::TextEncoder::load(
                    path,
                    4096,
                    encoder_device.clone(),
                    encoder_dtype,
                )?
            };
            let mut encode = |p: &str| -> Result<PromptEmbeds> {
                let (embeds, image_slots) = if images.is_empty() {
                    (encoder.encode(p)?, None)
                } else {
                    let (embeds, slots) = encoder.encode_with_images(p, images)?;
                    (embeds, Some(slots))
                };
                Ok(PromptEmbeds {
                    embeds: embeds.to_device(device)?,
                    image_slots,
                })
            };
            let negative = negative_prompt.map(&mut encode).transpose()?;
            (encode(prompt)?, negative)
        }
        None => {
            if !images.is_empty() {
                return Err(E::msg(
                    "--image needs --text-encoder-path (the vision tower is part of the text encoder)",
                ));
            }
            eprintln!("  No text encoder path, using random embeddings");
            let random = || -> Result<PromptEmbeds> {
                // f32: Metal has no F64 rand_uniform
                Ok(PromptEmbeds {
                    embeds: Tensor::randn(0.0f32, 1.0f32, (1, 256, 4096), device)?,
                    image_slots: None,
                })
            };
            (random()?, negative_prompt.map(|_| random()).transpose()?)
        }
    };
    release_freed_memory(device)?;
    eprintln!(
        "  Text encoding done in {:.2}s. Encoder released.",
        start.elapsed().as_secs_f32()
    );
    Ok(embs)
}

/// VAE-encodes the condition images into packed latents for the transformer.
pub fn encode_condition_images(
    vae_path: &str,
    images: &[crate::condition_image::ConditionImage],
    device: &Device,
) -> Result<ConditionLatents> {
    eprintln!(
        "[Phase 1b] Encoding {} condition image(s) with the VAE",
        images.len()
    );
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[vae_path.to_string()], DType::F32, device)?
    };
    let encoder = crate::vae::VaeEncoder::new(&VaeConfig::default(), vb)?;
    let mut packed = Vec::with_capacity(images.len());
    let mut shapes = Vec::with_capacity(images.len());
    for img in images {
        let latents = encoder.encode(&img.vae_input)?; // [1, 64, h, w]
        let (_, _, h, w) = latents.dims4()?;
        packed.push(crate::vae::pack_latents(&latents.unsqueeze(2)?)?);
        shapes.push((h, w));
    }
    let latents = Tensor::cat(&packed, 1)?;
    drop(encoder);
    release_freed_memory(device)?;
    Ok(ConditionLatents { latents, shapes })
}

/// Synchronize the device so freed memory is returned to the driver (CUDA's
/// stream-ordered allocator keeps freed memory reserved until synchronize).
pub fn release_freed_memory(device: &Device) -> Result<()> {
    Ok(device.synchronize()?)
}

/// Convert image tensor [-1,1] to PNG bytes. The VAE decoder outputs 4 channels
/// (RGBA), so the alpha channel is dropped before encoding as RGB PNG.
pub fn image_to_png_bytes(image: &Tensor, width: usize, height: usize) -> Result<Vec<u8>> {
    let image = image.clamp(-1.0, 1.0)?;
    let image = image.affine(0.5, 0.5)?; // (x + 1) * 0.5
    let image = image.affine(255.0, 0.0)?; // * 255
    let image = image.to_dtype(candle_core::DType::U8)?;
    let image = image.i(0)?; // [C, H, W]
    let image = image.permute((1, 2, 0))?; // [H, W, C]

    let channels = image.dim(2)?;
    let pixels = image.flatten_all()?.to_vec1::<u8>()?;
    let rgb: Vec<u8> = pixels
        .chunks(channels)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();

    let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_vec(width as u32, height as u32, rgb)
            .ok_or_else(|| E::msg("failed to create image buffer"))?;
    let dynamic = image::DynamicImage::ImageRgb8(img);
    let mut cursor = std::io::Cursor::new(Vec::new());
    dynamic.write_to(&mut cursor, image::ImageFormat::Png)?;
    Ok(cursor.into_inner())
}
