//! Diagnostic: compare each stage of the image-conditioned path against golden
//! tensors dumped by `tools/ref_dump_i2i.py` from the upstream diffusers pipeline.
//!
//! Usage: i2i_probe <stage> <ref_dir> [model_root]
//!   stages: vae, pre, vision, text, transformer, encode, loop
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

fn device() -> Result<Device> {
    Ok(match Device::cuda_if_available(0)? {
        Device::Cpu => Device::metal_if_available(0)?,
        d => d,
    })
}

fn load(ref_dir: &str, file: &str, key: &str, device: &Device) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(format!("{ref_dir}/{file}.safetensors"), device)?;
    match tensors.get(key) {
        Some(t) => Ok(t.to_dtype(DType::F32)?),
        None => bail!("{file}.safetensors has no `{key}` (keys: {:?})", tensors.keys().collect::<Vec<_>>()),
    }
}

/// Prints shape agreement, cosine similarity, and relative errors.
fn compare(label: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    if got.dims() != want.dims() {
        println!("{label}: SHAPE MISMATCH got {:?} want {:?}", got.dims(), want.dims());
        return Ok(());
    }
    let (g, w) = (got.flatten_all()?.to_dtype(DType::F64)?, want.flatten_all()?.to_dtype(DType::F64)?);
    let cos = (&g * &w)?.sum_all()?.to_scalar::<f64>()?
        / (g.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt() * w.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt());
    let diff = (&g - &w)?.abs()?;
    let mean_rel = diff.mean_all()?.to_scalar::<f64>()? / w.abs()?.mean_all()?.to_scalar::<f64>()?;
    let max_rel = diff.max_all()?.to_scalar::<f64>()? / w.abs()?.max_all()?.to_scalar::<f64>()?;
    println!("{label}: shape {:?}  cosine {cos:.8}  mean-rel {mean_rel:.3e}  max-rel {max_rel:.3e}", got.dims());
    Ok(())
}

fn text_encoder_vb(root: &str, dev: &Device) -> Result<VarBuilder<'static>> {
    let dir = format!("{root}/text_encoder");
    let mut paths: Vec<String> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path().to_string_lossy().to_string()))
        .filter(|p| p.ends_with(".safetensors"))
        .collect();
    paths.sort();
    let dt = match std::env::var("I2I_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        _ => DType::F32,
    };
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&paths, dt, dev)? })
}

fn load_vision(root: &str, dev: &Device) -> Result<qwen3_image21::qwen3_vl_vision::Qwen3VLVisionModel> {
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{root}/text_encoder/config.json"))?)?;
    let cfg: qwen3_image21::qwen3_vl_vision::VisionConfig = serde_json::from_value(cfg_json["vision_config"].clone())?;
    let vb = text_encoder_vb(root, dev)?;
    Ok(qwen3_image21::qwen3_vl_vision::Qwen3VLVisionModel::new(&cfg, vb.pp("model").pp("visual"))?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let stage = args.get(1).map(String::as_str).unwrap_or("vae");
    let ref_dir = args.get(2).cloned().unwrap_or_else(|| "/root/qi21_ref".into());
    let root = args.get(3).cloned().unwrap_or_else(|| "models/Qwen-Image-2.1-official".into());
    let dev = device()?;
    // I2I_DTYPE=bf16 loads the vision, text, and transformer stages in BF16 to
    // measure that precision against the (F32) golden tensors.
    let dt = match std::env::var("I2I_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        _ => DType::F32,
    };
    // I2I_TF32=1 allows TF32 for F32 matmuls (the CLI's --tf32).
    candle_core::cuda::set_gemm_reduced_precision_f32(std::env::var("I2I_TF32").as_deref() == Ok("1"));

    match stage {
        "vae" => {
            let image = load(&ref_dir, "image", "vae_image", &dev)?.squeeze(2)?; // [1,4,H,W]
            let want = load(&ref_dir, "vae", "image_latents", &dev)?.squeeze(2)?; // [1,64,h,w]
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[format!("{root}/vae/diffusion_pytorch_model.safetensors")], DType::F32, &dev)?
            };
            let encoder = qwen3_image21::vae::VaeEncoder::new(&qwen3_image21::vae::Config::default(), vb)?;
            compare("vae latents", &encoder.encode(&image)?, &want)?;
        }
        "pre" => {
            // Preprocessing from the exact resized pixels upstream used.
            let rgba = load(&ref_dir, "image", "resized_rgba", &Device::Cpu)?.to_dtype(DType::U8)?; // [H,W,4]
            let (h, w, _) = rgba.dims3()?;
            let img = image::RgbaImage::from_raw(w as u32, h as u32, rgba.flatten_all()?.to_vec1::<u8>()?).unwrap();
            let cond = qwen3_image21::condition_image::from_rgba(&img, &dev)?;
            compare("vae input", &cond.vae_input, &load(&ref_dir, "image", "vae_image", &dev)?.squeeze(2)?)?;
            compare("pixel_values", &cond.pixel_values, &load(&ref_dir, "processor", "pixel_values", &dev)?)?;
            if let Some(path) = args.get(4) {
                // Our own file loading (image crate) against upstream's PIL pixels.
                let (_, _, hh, ww) = cond.vae_input.dims4()?;
                let resolution = ((hh * ww) as f64).sqrt().round() as usize;
                let loaded = qwen3_image21::condition_image::load(path, resolution, &dev)?;
                compare("vae input (loaded from file)", &loaded.vae_input, &cond.vae_input)?;
            }
            println!("grid_thw: ours {:?}  upstream {:?}", cond.grid_thw.to_vec2::<u32>()?,
                load(&ref_dir, "processor", "image_grid_thw", &Device::Cpu)?.to_dtype(DType::U32)?.to_vec2::<u32>()?);
        }
        "vision" => {
            let vision = load_vision(&root, &dev)?;
            let pixel_values = load(&ref_dir, "processor", "pixel_values", &dev)?;
            let grid = load(&ref_dir, "processor", "image_grid_thw", &dev)?.to_dtype(DType::U32)?;
            let (embeds, deepstack) = vision.forward(&pixel_values, &grid)?;
            compare("image_embeds", &embeds, &load(&ref_dir, "vision", "image_embeds", &dev)?)?;
            for (i, d) in deepstack.iter().enumerate() {
                compare(&format!("deepstack_{i}"), d, &load(&ref_dir, "vision", &format!("deepstack_{i}"), &dev)?)?;
            }
        }
        "text" => {
            // Text encoder with upstream's token ids and vision features, so only
            // the language-model side (scatter, mRoPE, DeepStack) is under test.
            let dir = format!("{root}/text_encoder");
            let cfg_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
            let cfg: qwen3_image21::qwen3_vl_text::Config = serde_json::from_value(cfg_json.clone())?;
            let image_token_id = cfg_json["image_token_id"].as_u64().unwrap() as u32;
            let vb = text_encoder_vb(&root, &dev)?;
            let model = qwen3_image21::qwen3_vl_text::Qwen3VLTextEncoder::new(&cfg.text_config, vb)?;
            let input_ids = load(&ref_dir, "processor", "input_ids", &dev)?.to_dtype(DType::U32)?;
            let grid = load(&ref_dir, "processor", "image_grid_thw", &Device::Cpu)?.to_dtype(DType::U32)?.to_vec2::<u32>()?;
            let merge = cfg_json["vision_config"]["spatial_merge_size"].as_u64().unwrap() as usize;
            let features = qwen3_image21::qwen3_vl_text::ImageFeatures {
                embeds: load(&ref_dir, "vision", "image_embeds", &dev)?,
                deepstack: (0..3).map(|i| load(&ref_dir, "vision", &format!("deepstack_{i}"), &dev)).collect::<Result<_>>()?,
                merged_grids: grid.iter().map(|g| (g[0] as usize, g[1] as usize / merge, g[2] as usize / merge)).collect(),
                image_token_id,
            };
            let hidden = model.forward_with_images(&input_ids, Some(&features))?;
            let want = load(&ref_dir, "text", "prompt_embeds", &dev)?;
            let drop_idx = hidden.dim(1)? - want.dim(1)?;
            println!("dropping {drop_idx} system-prefix tokens");
            compare("prompt_embeds", &hidden.narrow(1, drop_idx, want.dim(1)?)?, &want)?;
        }
        "transformer" => {
            // One forward pass on upstream's exact inputs (no KV cache), then the
            // same with the cache (fill + reuse) to check the extended prefix.
            let prompt_embeds = load(&ref_dir, "text", "prompt_embeds", &dev)?;
            let slots: Vec<bool> = load(&ref_dir, "text", "image_pad_mask", &Device::Cpu)?
                .flatten_all()?.to_vec1::<f32>()?.iter().map(|&v| v > 0.5).collect();
            let latents = load(&ref_dir, "vae", "image_latents", &dev)?; // [1,64,1,h,w]
            let (_, _, _, h, w) = latents.dims5()?;
            let cond = qwen3_image21::vae::pack_latents(&latents)?;
            let noise = load(&ref_dir, "transformer", "noise", &dev)?;
            let sigma = load(&ref_dir, "transformer", "sigma", &Device::Cpu)?.to_vec1::<f32>()?[0];
            let t = qwen3_image21::scheduler::timestep_embedding(&Tensor::new(&[sigma], &dev)?, 256)?;
            let paths = qwen3_image21::safetensors_util::resolve_safetensors_paths(&format!(
                "{root}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
            ))?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dt, &dev)? };
            let model = qwen3_image21::transformer::QwenImageTransformer::new(&qwen3_image21::transformer::Config::default(), vb)?;
            let (prompt_embeds, cond, noise, t) = (prompt_embeds.to_dtype(dt)?, cond.to_dtype(dt)?, noise.to_dtype(dt)?, t.to_dtype(dt)?);
            let shapes = [(h, w)];
            let condition = qwen3_image21::joint_layout::ConditionTokens { latents: &cond, shapes: &shapes, text_image_mask: &slots };
            let (height, width) = (h * 16, w * 16); // target same size as the condition in the dump
            let want = load(&ref_dir, "transformer", "noise_pred", &dev)?;
            let pred = model.forward_conditioned(&noise, Some(&prompt_embeds), None, &t, height, width, Some(&condition), None)?;
            compare("noise_pred", &pred, &want)?;
            let mut cache = None;
            let first = model.forward_conditioned(&noise, Some(&prompt_embeds), None, &t, height, width, Some(&condition), Some(&mut cache))?;
            let reuse = model.forward_conditioned(&noise, Some(&prompt_embeds), None, &t, height, width, Some(&condition), Some(&mut cache))?;
            compare("noise_pred (cache fill)", &first, &want)?;
            compare("noise_pred (cache reuse)", &reuse, &want)?;
        }
        "encode" => {
            // Full prompt encoding from the image file: preprocessing, template +
            // tokenization, vision tower, and language model, all ours.
            // Uses the exact resized pixels upstream used (see `pre`), at their size.
            let rgba = load(&ref_dir, "image", "resized_rgba", &Device::Cpu)?.to_dtype(DType::U8)?;
            let (h, w, _) = rgba.dims3()?;
            let img = image::RgbaImage::from_raw(w as u32, h as u32, rgba.flatten_all()?.to_vec1::<u8>()?).unwrap();
            let cond = qwen3_image21::condition_image::from_rgba(&img, &dev)?;
            let mut encoder = qwen3_image21::text_encoder::TextEncoder::load(format!("{root}/text_encoder"), 4096, dev.clone(), candle_core::DType::F32)?;
            let (embeds, slots) = encoder.encode_with_images("Change the apple to a green apple", std::slice::from_ref(&cond))?;
            let want_slots: Vec<bool> = load(&ref_dir, "text", "image_pad_mask", &Device::Cpu)?
                .flatten_all()?.to_vec1::<f32>()?.iter().map(|&v| v > 0.5).collect();
            println!("image-slot mask: {} tokens, {} slots; matches upstream: {}", slots.len(), slots.iter().filter(|&&s| s).count(), slots == want_slots);
            compare("prompt_embeds", &embeds, &load(&ref_dir, "text", "prompt_embeds", &dev)?)?;
        }
        "loop" => {
            // The whole denoising loop from upstream's initial noise (with the
            // prefix KV cache, as the pipeline runs it), compared per step.
            let prompt_embeds = load(&ref_dir, "text", "prompt_embeds", &dev)?;
            let slots: Vec<bool> = load(&ref_dir, "text", "image_pad_mask", &Device::Cpu)?
                .flatten_all()?.to_vec1::<f32>()?.iter().map(|&v| v > 0.5).collect();
            let latents = load(&ref_dir, "vae", "image_latents", &dev)?;
            let (_, _, _, h, w) = latents.dims5()?;
            let cond = qwen3_image21::vae::pack_latents(&latents)?;
            let want_sigmas = load(&ref_dir, "loop", "sigmas", &Device::Cpu)?.to_vec1::<f32>()?;
            let steps = want_sigmas.len() - 1; // upstream appends a terminal 0
            let scheduler = qwen3_image21::scheduler::FlowMatchEuler::new(steps, h * w);
            let ours: Vec<f32> = scheduler.sigmas().to_vec();
            let max_dsigma = ours.iter().zip(&want_sigmas).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("sigmas: {} steps, max |ours - upstream| = {max_dsigma:.2e}", steps);
            let paths = qwen3_image21::safetensors_util::resolve_safetensors_paths(&format!(
                "{root}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
            ))?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &dev)? };
            let model = qwen3_image21::transformer::QwenImageTransformer::new(&qwen3_image21::transformer::Config::default(), vb)?;
            let shapes = [(h, w)];
            let condition = qwen3_image21::joint_layout::ConditionTokens { latents: &cond, shapes: &shapes, text_image_mask: &slots };
            let mut x = load(&ref_dir, "loop", "init", &dev)?;
            let mut cache = None;
            for i in 0..steps {
                let t = qwen3_image21::scheduler::timestep_embedding(&Tensor::new(&[ours[i]], &dev)?, 256)?;
                let pred = model.forward_conditioned(&x, Some(&prompt_embeds), None, &t, h * 16, w * 16, Some(&condition), Some(&mut cache))?;
                x = scheduler.step(&pred, &x, i)?;
                compare(&format!("latents after step {i}"), &x, &load(&ref_dir, "loop", &format!("step_{i}"), &dev)?)?;
            }
        }
        "noise" => {
            // Statistics of the initial noise exactly as pipeline::denoise draws it.
            let n = Tensor::randn(0f32, 1f32, (1, 64, 1, 64, 64), &dev)?;
            let flat = n.flatten_all()?;
            let mean = flat.mean_all()?.to_scalar::<f32>()?;
            let std = (flat.broadcast_sub(&Tensor::new(mean, &dev)?)?.sqr()?.mean_all()?.to_scalar::<f32>()?).sqrt();
            let v: Vec<f32> = flat.to_vec1()?;
            let frac_over_3 = v.iter().filter(|x| x.abs() > 3.0).count() as f32 / v.len() as f32;
            println!("candle randn on {:?}: mean {mean:.4} std {std:.4} |x|>3 fraction {frac_over_3:.5} (N(0,1): 0.00270); first values {:?}", dev, &v[..6]);
            if let Some(path) = args.get(3) {
                let packed = qwen3_image21::vae::pack_latents(&n)?;
                candle_core::safetensors::save(&std::collections::HashMap::from([("noise".to_string(), packed)]), path)?;
                println!("saved packed noise to {path}");
            }
        }
        other => bail!("unknown stage `{other}`"),
    }
    Ok(())
}
