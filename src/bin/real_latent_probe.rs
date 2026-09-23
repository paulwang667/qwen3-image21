//! Diagnostic: decode a REAL packed_latents tensor dumped from an actual
//! denoising run (via QWEN_DUMP_LATENTS_PATH), and measure its per-position
//! correlation the same way transformer_probe does — to place it on the
//! decorrelated-noise <-> degenerate-uniform spectrum vae_probe established,
//! and confirm (or rule out) that the checkerboard artifact traces to the
//! VAE's sensitivity to that correlation rather than a separate bug.
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use qwen3_image21::vae::{normalize_latents, unpack_latents, Config as VaeConfig, VaeDecoder};

fn mean_pairwise_cosine(x: &Tensor, max_pairs: usize) -> Result<f64> {
    let (n, _d) = x.dims2()?;
    let norm = x.sqr()?.sum(1)?.sqrt()?;
    let normed = x.broadcast_div(&norm.unsqueeze(1)?)?;
    let sim = normed.matmul(&normed.transpose(0, 1)?)?;
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

fn main() -> Result<()> {
    let device = Device::Cpu;
    let latents_path = std::env::args().nth(1).unwrap_or_else(|| "/root/real_latents.safetensors".to_string());
    let vae_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/Qwen-Image-2.1-official/vae/diffusion_pytorch_model.safetensors".to_string());
    let height: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let width: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(256);

    let tensors = candle_core::safetensors::load(&latents_path, &device)?;
    let packed_latents = tensors.get("packed_latents").expect("missing packed_latents key").clone();
    println!("packed_latents.shape = {:?}", packed_latents.shape());

    // Correlation stats directly on the packed (pre-VAE) latent, same metric
    // transformer_probe used: [seq, channels] rows, pairwise cosine similarity.
    let (_b, seq, c) = packed_latents.dims3()?;
    let flat = packed_latents.reshape((seq, c))?;
    let mean = flat.mean(0)?;
    let var = flat.broadcast_sub(&mean)?.sqr()?.mean(0)?.mean(0)?.to_scalar::<f32>()?;
    let cos = mean_pairwise_cosine(&flat, 500)?;
    println!("real packed_latents: n={seq} d={c} per-channel-var(mean)={var:.6} mean-pairwise-cosine={cos:.4}");

    // Decode it exactly like the real pipeline does.
    let cfg = VaeConfig::default();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path], DType::F32, &device)? };
    let decoder = VaeDecoder::new(&cfg, vb)?;

    let scale = cfg.spatial_compression_ratio();
    let latents = unpack_latents(&packed_latents, height, width, scale)?;
    let latents = normalize_latents(&latents, &cfg.latents_mean, &cfg.latents_std)?;
    let latents = latents.squeeze(2)?;
    let image = decoder.decode(&latents)?;

    let image = image.clamp(-1.0, 1.0)?.affine(0.5, 0.5)?.affine(255.0, 0.0)?.to_dtype(DType::U8)?;
    let image = image.i(0)?.permute((1, 2, 0))?;
    let (out_h, out_w, out_c) = image.dims3()?;
    let pixels = image.flatten_all()?.to_vec1::<u8>()?;
    let rgb: Vec<u8> = pixels.chunks(out_c).flat_map(|px| [px[0], px[1], px[2]]).collect();
    let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_vec(out_w as u32, out_h as u32, rgb).unwrap();
    img.save("/tmp/real_latent_decode.png")?;
    println!("Saved /tmp/real_latent_decode.png");

    Ok(())
}
