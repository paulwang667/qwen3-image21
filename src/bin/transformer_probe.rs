//! Diagnostic: run a single quantized-transformer forward pass on random inputs and
//! measure how "diverse" the per-position (per image-token) output is, to check
//! whether the transformer is collapsing distinct spatial positions toward the same
//! output (which would explain the checkerboard-prone, spatially-degenerate latents
//! that `vae_probe` showed trigger the VAE's depth-to-space tiling artifact).
use anyhow::Result;
use candle_core::{Device, Tensor};
use qwen3_image21::gguf_mapped::load_gguf_mapped;
use qwen3_image21::quantized_transformer::QwenImageTransformerQuantized;
use qwen3_image21::transformer::Config;

/// Mean pairwise cosine similarity between rows of `x` ([N, D]), sampling up to
/// `max_pairs` pairs. 1.0 = every position identical, ~0 = orthogonal/diverse.
fn mean_pairwise_cosine(x: &Tensor, max_pairs: usize) -> Result<f64> {
    let (n, _d) = x.dims2()?;
    let norm = x.sqr()?.sum(1)?.sqrt()?; // [N]
    let normed = x.broadcast_div(&norm.unsqueeze(1)?)?; // [N, D]
    let sim = normed.matmul(&normed.transpose(0, 1)?)?; // [N, N]
    let sim: Vec<f32> = sim.flatten_all()?.to_vec1()?;
    let mut total = 0.0f64;
    let mut count = 0usize;
    let mut i = 0usize;
    let mut j = 1usize;
    while count < max_pairs && i < n {
        if j >= n {
            i += 1;
            j = i + 1;
            continue;
        }
        total += sim[i * n + j] as f64;
        count += 1;
        j += (n / max_pairs.max(1)).max(1);
    }
    Ok(if count > 0 { total / count as f64 } else { 0.0 })
}

fn stats(name: &str, x: &Tensor) -> Result<()> {
    let (n, d) = x.dims2()?;
    let mean = x.mean(0)?; // [D]
    let var = x.broadcast_sub(&mean)?.sqr()?.mean(0)?.mean(0)?.to_scalar::<f32>()?;
    let cos = mean_pairwise_cosine(x, 500)?;
    println!("{name}: n={n} d={d} per-channel-var(mean)={var:.6} mean-pairwise-cosine={cos:.4}");
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let cfg = Config::default();

    let mapped_vb = load_gguf_mapped("models/qwen-image-2.1-Q4_0.gguf", &device)?;
    let transformer = QwenImageTransformerQuantized::new(&cfg, mapped_vb.inner().clone())?;

    let height = 256usize;
    let width = 256usize;
    let seq_img = 256usize; // 16x16 at vae_scale_factor=16
    let seq_txt = 256usize;

    let hidden_states = Tensor::randn(0.0f32, 1.0f32, (1, seq_img, cfg.in_channels), &device)?;
    let text_emb = Tensor::randn(0.0f32, 1.0f32, (1, seq_txt, cfg.joint_attention_dim), &device)?;
    let timestep = Tensor::new(&[0.5f32], &device)?; // mid-schedule timestep, [B]
    let timestep = qwen3_image21::scheduler::timestep_embedding(&timestep, 256)?;
    let dummy_ids = Tensor::zeros((1, 1, 2), candle_core::DType::F32, &device)?;

    println!("Running single forward pass...");
    let noise_pred = transformer.forward(
        &hidden_states, Some(&text_emb), None, &timestep, &dummy_ids, &dummy_ids, height, width,
    )?;
    println!("noise_pred shape: {:?}", noise_pred.shape());

    let hs2d = hidden_states.reshape((seq_img, cfg.in_channels))?;
    let np2d = noise_pred.reshape((seq_img, cfg.out_channels))?;

    stats("input hidden_states (random, baseline)", &hs2d)?;
    stats("transformer noise_pred (image tokens)", &np2d)?;

    Ok(())
}
