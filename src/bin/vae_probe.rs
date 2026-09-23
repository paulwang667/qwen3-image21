//! Diagnostic: decode a smooth synthetic latent (no transformer involved) to check
//! whether the VAE decoder itself produces checkerboard/grid artifacts independent
//! of denoising content.
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use qwen3_image21::vae::{Config as VaeConfig, VaeDecoder};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let cfg = VaeConfig::default();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["models/vae/qwen_image_2.1_vae_bf16.safetensors".to_string()],
            DType::F32,
            &device,
        )?
    };
    let decoder = VaeDecoder::new(&cfg, vb)?;

    let h = 16usize;
    let w = 16usize;

    let save = |latents: &Tensor, name: &str| -> Result<()> {
        let image = decoder.decode(latents)?;
        let image = image.clamp(-1.0, 1.0)?.affine(0.5, 0.5)?.affine(255.0, 0.0)?;
        let image = image.to_dtype(DType::U8)?;
        let image = image.i(0)?.permute((1, 2, 0))?;
        let (out_h, out_w, out_c) = image.dims3()?;
        let pixels = image.flatten_all()?.to_vec1::<u8>()?;
        let rgb: Vec<u8> = pixels.chunks(out_c).flat_map(|px| [px[0], px[1], px[2]]).collect();
        let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_vec(out_w as u32, out_h as u32, rgb).unwrap();
        img.save(name)?;
        println!("Saved {name}");
        Ok(())
    };

    // Test 1: all-zeros latent (isolates any texture coming purely from conv/norm biases).
    let zeros = Tensor::zeros((1, cfg.z_dim, h, w), DType::F32, &device)?;
    save(&zeros, "/tmp/vae_probe_zeros.png")?;

    // Test 2: smooth low-frequency gradient, same across all 64 channels.
    let mut data = vec![0f32; cfg.z_dim * h * w];
    for c in 0..cfg.z_dim {
        for y in 0..h {
            for x in 0..w {
                let v = (x as f32 / w as f32 * std::f32::consts::PI).sin()
                    + (y as f32 / h as f32 * std::f32::consts::PI).cos();
                data[c * h * w + y * w + x] = v;
            }
        }
    }
    let smooth = Tensor::from_vec(data, (1, cfg.z_dim, h, w), &device)?;
    save(&smooth, "/tmp/vae_probe_smooth.png")?;

    // Test 3: realistic per-channel-independent random noise, denormalized with the
    // real latents_mean/std (matching what an early, high-noise denoising step hands
    // to the VAE — a fair stand-in for the pipeline's actual input distribution).
    let noise = Tensor::randn(0.0f32, 1.0f32, (1, cfg.z_dim, h, w), &device)?;
    let mean = Tensor::new(cfg.latents_mean.as_slice(), &device)?.reshape((1, cfg.z_dim, 1, 1))?;
    let std = Tensor::new(cfg.latents_std.as_slice(), &device)?.reshape((1, cfg.z_dim, 1, 1))?;
    let realistic = noise.broadcast_mul(&std)?.broadcast_add(&mean)?;
    save(&realistic, "/tmp/vae_probe_noise.png")?;

    Ok(())
}
