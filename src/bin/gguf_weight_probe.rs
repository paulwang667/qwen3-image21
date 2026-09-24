//! Diagnostic: dequantize selected GGUF transformer tensors and diff them
//! against the matching official safetensors weights, to tell a weight
//! layout/dtype problem apart from a forward-pass bug in the quantized path.
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};

fn stats(label: &str, gguf: &Tensor, reference: &Tensor) -> Result<()> {
    if gguf.dims() != reference.dims() {
        println!("{label}: SHAPE MISMATCH gguf {:?} vs official {:?}", gguf.dims(), reference.dims());
        return Ok(());
    }
    let diff = (gguf - reference)?.abs()?;
    let max = diff.max_all()?.to_scalar::<f32>()?;
    let mean = diff.mean_all()?.to_scalar::<f32>()?;
    let ref_mean = reference.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("{label}: shape {:?}  mean|diff| {mean:.3e}  max|diff| {max:.3e}  mean|ref| {ref_mean:.3e}  rel {:.3e}", gguf.dims(), mean / ref_mean);
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let gguf_path = std::env::args().nth(1).unwrap_or_else(|| "models/qwen-image-2.1-Q8_0.gguf".into());
    let root = std::env::args().nth(2).unwrap_or_else(|| "models/Qwen-Image-2.1-official".into());

    let mut file = std::fs::File::open(&gguf_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let shards = qwen3_image21::safetensors_util::resolve_safetensors_paths(&format!(
        "{root}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
    ))?;
    let st = unsafe { MmapedSafetensors::multi(&shards)? };

    let mut gguf = |name: &str| -> Result<Tensor> {
        let q = content.tensor(&mut file, &format!("model.diffusion_model.{name}"), &device)?;
        println!("  [{name}] ggml dtype {:?}", q.dtype());
        Ok(q.dequantize(&device)?.to_dtype(DType::F32)?)
    };
    let official = |name: &str| -> Result<Tensor> { Ok(st.load(name, &device)?.to_dtype(DType::F32)?) };

    for name in [
        "img_in.weight",
        "txt_in.text_norm.weight",
        "txt_in.in_layer.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "modulation.1.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.img_mlp.out.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    ] {
        stats(name, &gguf(name)?, &official(name)?)?;
    }

    let gate_up = gguf("transformer_blocks.0.img_mlp.gate_up.weight")?;
    let gate_layer = official("transformer_blocks.0.img_mlp.gate_layer.weight")?;
    let proj = official("transformer_blocks.0.img_mlp.proj.weight")?;
    stats("gate_up vs cat(gate_layer, proj)", &gate_up, &Tensor::cat(&[&gate_layer, &proj], 0)?)?;
    stats("gate_up vs cat(proj, gate_layer)", &gate_up, &Tensor::cat(&[&proj, &gate_layer], 0)?)?;
    Ok(())
}
