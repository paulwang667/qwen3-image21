//! Condition-image preprocessing for image-conditioned generation, matching the
//! upstream `QwenImage21Pipeline`: one resize (to ~`resolution²` pixels at the
//! image's aspect ratio, multiples of 32) feeds both the VAE (RGBA in [-1, 1])
//! and the Qwen3-VL vision tower (RGB over white, patchified like
//! `Qwen2VLImageProcessor`).
use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{imageops::FilterType, RgbaImage};

/// Vision patch size, spatial merge size, and temporal patch size of the
/// checkpoint's `processor/preprocessor_config.json`.
const PATCH: usize = 16;
const MERGE: usize = 2;
const TEMPORAL_PATCH: usize = 2;

pub struct ConditionImage {
    pub width: usize,
    pub height: usize,
    /// `[1, 4, H, W]` RGBA in [-1, 1] for the VAE encoder.
    pub vae_input: Tensor,
    /// `[grid_h * grid_w, 3 * 2 * 16 * 16]` for the vision tower.
    pub pixel_values: Tensor,
    /// `[1, 3]` = `(1, grid_h, grid_w)` in 16-px patches.
    pub grid_thw: Tensor,
}

impl ConditionImage {
    /// Number of `<|image_pad|>` tokens: one per merged 32x32-px vision patch
    /// (equivalently, one per 2x2 group of 16-px latent cells).
    pub fn num_image_tokens(&self) -> usize {
        (self.height / PATCH / MERGE) * (self.width / PATCH / MERGE)
    }
}

/// Upstream `calculate_dimensions`: `target_area` at `ratio` (w / h), rounded
/// to multiples of 32.
pub fn calculate_dimensions(target_area: f64, ratio: f64) -> (usize, usize) {
    let width = (target_area * ratio).sqrt();
    let height = width / ratio;
    let round32 = |v: f64| ((v / 32.0).round() as usize) * 32;
    (round32(width), round32(height))
}

pub fn load(path: &str, resolution: usize, device: &Device) -> Result<ConditionImage> {
    let img = image::open(path)?.to_rgba8();
    let (w, h) = calculate_dimensions((resolution * resolution) as f64, img.width() as f64 / img.height() as f64);
    // Upstream resizes with PIL Lanczos; image's Lanczos3 is close but not
    // bit-identical, so exact comparisons should use an already-sized input.
    let img = if (img.width() as usize, img.height() as usize) == (w, h) {
        img
    } else {
        image::imageops::resize(&img, w as u32, h as u32, FilterType::Lanczos3)
    };
    from_rgba(&img, device)
}

/// Load a condition image from raw bytes (e.g. an HTTP upload), same pipeline
/// as `load` but without touching disk.
pub fn load_from_bytes(bytes: &[u8], resolution: usize, device: &Device) -> Result<ConditionImage> {
    let img = image::load_from_memory(bytes)?.to_rgba8();
    let (w, h) = calculate_dimensions((resolution * resolution) as f64, img.width() as f64 / img.height() as f64);
    let img = if (img.width() as usize, img.height() as usize) == (w, h) {
        img
    } else {
        image::imageops::resize(&img, w as u32, h as u32, FilterType::Lanczos3)
    };
    from_rgba(&img, device)
}

pub fn from_rgba(img: &RgbaImage, device: &Device) -> Result<ConditionImage> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    anyhow::ensure!(w % (PATCH * MERGE) == 0 && h % (PATCH * MERGE) == 0, "condition image {w}x{h} is not a multiple of 32");
    let px = img.as_raw();

    // VAE: all four channels, [0, 255] -> [-1, 1], channel-first.
    let mut vae = vec![0f32; 4 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                vae[(c * h + y) * w + x] = px[(y * w + x) * 4 + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    let vae_input = Tensor::from_vec(vae, (1, 4, h, w), device)?;

    // Vision: alpha composited over white (as the checkpoint was trained), then
    // rescale by 1/255 and normalize with mean = std = 0.5, i.e. [-1, 1].
    let rgb = |y: usize, x: usize, c: usize| -> f32 {
        let a = px[(y * w + x) * 4 + 3] as f32 / 255.0;
        let v = px[(y * w + x) * 4 + c] as f32 * a + 255.0 * (1.0 - a);
        (v.round() / 255.0 - 0.5) / 0.5
    };
    // Qwen2VLImageProcessor patch order: (grid_t, gh/m, gw/m, m_h, m_w) patches,
    // each flattened as (C, temporal, ph, pw); one image is repeated to fill
    // the temporal patch.
    let (gh, gw) = (h / PATCH, w / PATCH);
    let patch_len = 3 * TEMPORAL_PATCH * PATCH * PATCH;
    let mut pv = Vec::with_capacity(gh * gw * patch_len);
    for bh in 0..gh / MERGE {
        for bw in 0..gw / MERGE {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    let (py, px0) = ((bh * MERGE + mh) * PATCH, (bw * MERGE + mw) * PATCH);
                    for c in 0..3 {
                        for _t in 0..TEMPORAL_PATCH {
                            for iy in 0..PATCH {
                                for ix in 0..PATCH {
                                    pv.push(rgb(py + iy, px0 + ix, c));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let pixel_values = Tensor::from_vec(pv, (gh * gw, patch_len), device)?;
    let grid_thw = Tensor::new(&[[1u32, gh as u32, gw as u32]], device)?;
    Ok(ConditionImage { width: w, height: h, vae_input, pixel_values, grid_thw })
}
