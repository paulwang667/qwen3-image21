//! Diagnostic: check that the text-prefix K/V cache is exact. Step 1 fills the
//! cache (full sequence), step 2 reuses it (image tokens only) at a different
//! timestep and latents; both are compared against the uncached forward pass.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use qwen3_image21::pipeline::TransformerType;
use qwen3_image21::quantized_transformer::QwenImageTransformerQuantized;
use qwen3_image21::transformer::{Config, QwenImageTransformer};

fn max_rel_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let diff = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    Ok(diff / b.abs()?.max_all()?.to_scalar::<f32>()?)
}

fn mean_rel_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let diff = (a - b)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    Ok(diff / b.abs()?.mean_all()?.to_scalar::<f32>()?)
}

fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let (a, b) = (a.flatten_all()?.to_dtype(DType::F64)?, b.flatten_all()?.to_dtype(DType::F64)?);
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f64>()?;
    Ok(dot / (a.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt() * b.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt()))
}

fn main() -> Result<()> {
    let device = match std::env::args().nth(2).as_deref() {
        Some("cpu") => Device::Cpu,
        _ => match Device::cuda_if_available(0)? {
            Device::Cpu => Device::metal_if_available(0)?,
            d => d,
        },
    };
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "models/Qwen-Image-2.1-official/transformer/diffusion_pytorch_model-00001-of-00002.safetensors".into()
    });
    let size: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let cfg = Config::default();
    let model = if model_path.ends_with(".gguf") {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&model_path, &device)?;
        let vb = if vb.contains_key("model.diffusion_model.img_in.weight") { vb.pp("model.diffusion_model") } else { vb };
        TransformerType::Quantized(QwenImageTransformerQuantized::new(&cfg, vb)?)
    } else {
        let paths = qwen3_image21::safetensors_util::resolve_safetensors_paths(&model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &device)? };
        TransformerType::NonQuantized(QwenImageTransformer::new(&cfg, vb)?)
    };

    let tokens = (size / 16) * (size / 16);
    let text = Tensor::randn(0f32, 1f32, (1, 20, 4096), &device)?;
    let dummy = Tensor::zeros((1, 1, 2), DType::F32, &device)?;
    let mut cache = None;
    for (step, sigma) in [(1, 0.9f32), (2, 0.4f32)] {
        let latents = Tensor::randn(0f32, 1f32, (1, tokens, 64), &device)?;
        let t = qwen3_image21::scheduler::timestep_embedding(&Tensor::new(&[sigma], &device)?, 256)?;
        let uncached = model.forward(&latents, &t, &text, None, &dummy, &dummy, size, size)?;
        let cached = model.forward_with_text_cache(&latents, &t, &text, None, size, size, &mut cache)?;
        println!(
            "step {step} ({}): shape {:?}  max-rel {:.3e}  mean-rel {:.3e}  cosine {:.8}",
            if step == 1 { "fills cache" } else { "reuses cache" },
            cached.dims(),
            max_rel_diff(&cached, &uncached)?,
            mean_rel_diff(&cached, &uncached)?,
            cosine(&cached, &uncached)?
        );
        // Run-to-run noise floor: the same uncached call twice.
        let again = model.forward(&latents, &t, &text, None, &dummy, &dummy, size, size)?;
        println!("        uncached vs uncached rerun: max-rel {:.3e}", max_rel_diff(&again, &uncached)?);
    }
    Ok(())
}
