//! Diagnostic: one transformer forward pass on identical inputs through the
//! full-precision safetensors model (CUDA/Metal), the quantized GGUF model on
//! the same device, and the quantized GGUF model on CPU. Separates a bug in the
//! quantized forward code from a device-specific quantized-kernel problem.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use qwen3_image21::pipeline::TransformerType;
use qwen3_image21::quantized_transformer::QwenImageTransformerQuantized;
use qwen3_image21::transformer::{Config, QwenImageTransformer};

fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let (a, b) = (a.flatten_all()?.to_dtype(DType::F64)?, b.flatten_all()?.to_dtype(DType::F64)?);
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f64>()?;
    Ok(dot / (a.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt() * b.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt()))
}

fn run(model: &TransformerType, inputs: &[Tensor; 3], device: &Device, size: usize) -> Result<Tensor> {
    let [latents, text, timestep] = inputs;
    let dummy = Tensor::zeros((1, 1, 2), DType::F32, device)?;
    let out = model.forward(
        &latents.to_device(device)?, &timestep.to_device(device)?, &text.to_device(device)?,
        None, &dummy, &dummy, size, size,
    )?;
    Ok(out.to_device(&Device::Cpu)?)
}

fn main() -> Result<()> {
    let gpu = match Device::cuda_if_available(0)? {
        Device::Cpu => Device::metal_if_available(0)?,
        d => d,
    };
    let root = "models/Qwen-Image-2.1-official";
    let gguf = std::env::args().nth(1).unwrap_or_else(|| "models/qwen-image-2.1-Q8_0.gguf".into());
    let size = 256usize; // 16x16 latent tokens: small enough for a CPU pass
    let cfg = Config::default();

    let cpu = Device::Cpu;
    let inputs = [
        Tensor::randn(0f32, 1f32, (1, (size / 16) * (size / 16), 64), &cpu)?,
        Tensor::randn(0f32, 1f32, (1, 12, 4096), &cpu)?,
        qwen3_image21::scheduler::timestep_embedding(&Tensor::new(&[0.5f32], &cpu)?, 256)?,
    ];

    let full = {
        let paths = qwen3_image21::safetensors_util::resolve_safetensors_paths(&format!(
            "{root}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
        ))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &gpu)? };
        let model = TransformerType::NonQuantized(QwenImageTransformer::new(&cfg, vb)?);
        run(&model, &inputs, &gpu, size)?
    };

    let load_quantized = |device: &Device| -> Result<TransformerType> {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&gguf, device)?;
        let vb = if vb.contains_key("model.diffusion_model.img_in.weight") { vb.pp("model.diffusion_model") } else { vb };
        Ok(TransformerType::Quantized(QwenImageTransformerQuantized::new(&cfg, vb)?))
    };
    let quant_gpu = run(&load_quantized(&gpu)?, &inputs, &gpu, size)?;
    let quant_cpu = run(&load_quantized(&cpu)?, &inputs, &cpu, size)?;

    println!("cosine(full, quant@gpu) = {:.6}", cosine(&full, &quant_gpu)?);
    println!("cosine(full, quant@cpu) = {:.6}", cosine(&full, &quant_cpu)?);
    println!("cosine(quant@gpu, quant@cpu) = {:.6}", cosine(&quant_gpu, &quant_cpu)?);
    Ok(())
}
