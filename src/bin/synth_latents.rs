//! Diagnostic: generate two synthetic packed_latents.safetensors files at the two
//! extremes of the correlation spectrum vae_probe/transformer_probe established —
//! fully decorrelated per-position noise, and fully degenerate (every position
//! identical) — so real_latent_probe can be run on them with the exact same
//! metric as the real denoised latents, giving an apples-to-apples reference.
use anyhow::Result;
use candle_core::{Device, Tensor};
use qwen3_image21::vae::Config as VaeConfig;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let cfg = VaeConfig::default();
    let seq = 256usize; // 16x16 at vae_scale_factor=16, matches the real 256x256 run
    let d = cfg.z_dim;

    let mean = Tensor::new(cfg.latents_mean.as_slice(), &device)?.reshape((1, 1, d))?;
    let std = Tensor::new(cfg.latents_std.as_slice(), &device)?.reshape((1, 1, d))?;

    // Decorrelated: independent random noise per position, denormalized to the
    // real latent distribution's scale.
    let noise = Tensor::randn(0.0f32, 1.0f32, (1, seq, d), &device)?;
    let decorrelated = noise.broadcast_mul(&std)?.broadcast_add(&mean)?;
    candle_core::safetensors::save(
        &std::collections::HashMap::from([("packed_latents".to_string(), decorrelated)]),
        "/tmp/synth_decorrelated.safetensors",
    )?;
    println!("Saved /tmp/synth_decorrelated.safetensors");

    // Degenerate: one random row broadcast to every position (fully uniform
    // across the spatial grid).
    let one_row = Tensor::randn(0.0f32, 1.0f32, (1, 1, d), &device)?;
    let one_row = one_row.broadcast_mul(&std)?.broadcast_add(&mean)?;
    let degenerate = one_row.broadcast_as((1, seq, d))?.contiguous()?;
    candle_core::safetensors::save(
        &std::collections::HashMap::from([("packed_latents".to_string(), degenerate)]),
        "/tmp/synth_degenerate.safetensors",
    )?;
    println!("Saved /tmp/synth_degenerate.safetensors");

    Ok(())
}
