//! Qwen-Image-2.1 3-axis (frame, height, width) Rotary Positional Embedding
//!
//! Matches diffusers `QwenImage21Rope` implementation.
//! Text tokens advance on all three axes. Image blocks freeze frame axis at the
//! position reached by preceding text and lay out tokens on a height/width grid
//! centered on zero.

use candle_core::{Result, Tensor, Device, DType, IndexOp};

/// RoPE computation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeMode {
    /// Precomputed cos/sin (default, matches diffusers use_real=True)
    Precomputed,
    /// Runtime cos/sin from angles (matches diffusers use_real=False / neuron path)
    Complex,
}

/// 3-axis RoPE for Qwen-Image-2.1
#[derive(Debug, Clone)]
pub struct EmbedNd {
    /// Precomputed complex frequencies for each axis: [freq_len, half, 2] complex as [real, imag]
    freqs: [Tensor; 3],
    /// Precomputed raw angles for each axis: [freq_len, half] (for Complex mode)
    angles: Option<[Tensor; 3]>,
    axes_dim: [usize; 3],
    mode: RopeMode,
}
impl EmbedNd {
    /// Create new EmbedNd with precomputed frequency tables.
    /// Matches diffusers: pos_index=8192, neg_index=1024
    pub fn new(_dim: usize, theta: f64, axes_dim: [usize; 3], device: &Device) -> Result<Self> {
        Self::new_with_mode(_dim, theta, axes_dim, device, RopeMode::Precomputed)
    }

    /// Create new EmbedNd with specified RoPE mode.
    pub fn new_with_mode(_dim: usize, theta: f64, axes_dim: [usize; 3], device: &Device, mode: RopeMode) -> Result<Self> {
        let theta = theta as f32;

        // Larger frequency tables to handle longer sequences
        // pos_index: 0..8191, neg_index: -1024..-1 (flipped)
        // Explicit i64: untyped literals default to i32, and Metal has no I32->F32 cast kernel.
        let pos_index = Tensor::arange(0i64, 8192i64, device)?.to_dtype(DType::F32)?;
        let neg_index = Tensor::arange(1i64, 1025i64, device)?
            .to_dtype(DType::F32)?
            .flip(&[0])?
            .affine(-1.0, 0.0)?; // -1024, -1023, ..., -1

        let mut freqs = Vec::with_capacity(3);
        let mut angles = Vec::with_capacity(3);
        for &dim in &axes_dim {
            // freq = theta^(-2i/dim) for i in 0..dim/2, i.e. exp(i * (-2*ln(theta)/dim))
            let half = dim / 2;
            let ln_theta = (theta as f64).ln();
            let inv_freq = Tensor::arange(0, half as i64, device)?
                .to_dtype(DType::F32)?
                .affine(-2.0 * ln_theta / dim as f64, 0.0)?
                .exp()?; // theta^(-2i/dim)

            // pos_freqs = outer(pos_index, inv_freq)
            let pos_outer = pos_index.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [8192, half]
            // neg_freqs = outer(neg_index, inv_freq)
            let neg_outer = neg_index.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [1024, half]

            // Concat angles: [neg, pos] so index 0 = -1024, index 1024 = 0
            let combined_angles = Tensor::cat(&[&neg_outer, &pos_outer], 0)?; // [9216, half]
            angles.push(combined_angles);

            // polar: complex(cos, sin) = exp(i * x)
            let pos_complex = Self::polar(&pos_outer)?; // [8192, half]
            let neg_complex = Self::polar(&neg_outer)?; // [1024, half]

            // Concat: [neg, pos] so index 0 = -1024, index 1024 = 0
            let combined = Tensor::cat(&[&neg_complex, &pos_complex], 0)?; // [9216, half]
            freqs.push(combined);
        }

        Ok(Self {
            freqs: [freqs[0].clone(), freqs[1].clone(), freqs[2].clone()],
            angles: if mode == RopeMode::Complex {
                Some([angles[0].clone(), angles[1].clone(), angles[2].clone()])
            } else {
                None
            },
            axes_dim,
            mode,
        })
    }

    /// Compute complex polar: exp(i * x) = cos(x) + i*sin(x)
    /// Returns [n, half, 2] where last dim is [real, imag]
    fn polar(x: &Tensor) -> Result<Tensor> {
        let cos = x.cos()?;
        let sin = x.sin()?;
        Tensor::cat(&[&cos.unsqueeze(2)?, &sin.unsqueeze(2)?], 2)
    }

    /// Forward: compute RoPE frequencies for joint text/image sequence.
    ///
    /// Args:
    ///   - `img_shapes`: list of (frame, height, width) per image block (condition first, target last)
    ///   - `image_pad_mask`: [seq_len] bool, true at image token positions
    /// Returns: [seq_len, sum(axes_dim/2)*2] = [seq_len, hidden_size] complex as [real, imag] interleaved
    pub fn forward(
        &self,
        img_shapes: &[(usize, usize, usize)],
        image_pad_mask: &Tensor,
        _device: &Device,
    ) -> Result<Tensor> {
        let seq_len = image_pad_mask.dim(0)?;
        let is_image_token: Vec<bool> = (0..seq_len)
            .map(|i| {
                let v = image_pad_mask.get(i).unwrap();
                v.to_scalar::<u8>().unwrap() != 0
            })
            .collect();

        let mut frame_index = Vec::with_capacity(seq_len);
        let mut height_index = Vec::with_capacity(seq_len);
        let mut width_index = Vec::with_capacity(seq_len);

        let mut cursor = 0usize;
        let mut position = 0i64;

        for (block_idx, &(frame, height, width)) in img_shapes.iter().enumerate() {
            let block_tokens = frame * height * width;

            // Find text tokens before this image block
            let block_start = is_image_token[cursor..]
                .iter()
                .position(|&x| x)
                .map(|p| cursor + p)
                .unwrap_or(seq_len);
            let text_len = block_start - cursor;

            // Text tokens: advance position on all axes
            for _ in 0..text_len {
                frame_index.push(position);
                height_index.push(position);
                width_index.push(position);
                position += 1;
            }
            cursor = block_start;

            // Image block tokens: freeze frame at current position, lay out height/width grid centered on zero
            let block_tokens = frame * height * width;
            for _ in 0..block_tokens {
                frame_index.push(position);
            }
            position += (height.max(width)) as i64;

            // Height/width grid centered on zero
            let h_start = -(height as i64 - height as i64 / 2);
            let w_start = -(width as i64 - width as i64 / 2);
            for h in 0..height {
                for w in 0..width {
                    height_index.push(h_start + h as i64);
                    width_index.push(w_start + w as i64);
                }
            }
            cursor += block_tokens;
        }

        // Remaining text after last image block
        if cursor < seq_len {
            for _ in cursor..seq_len {
                frame_index.push(position);
                height_index.push(position);
                width_index.push(position);
                position += 1;
            }
        }

        // Convert to tensors. Table layout is [neg(-1024..-1), pos(0..8191)], so raw
        // position `p` lives at table index `p + 1024` (e.g. p=-8, from the image
        // grid centered on zero, maps to index 1016, not to a negative/wrapped index).
        const TABLE_OFFSET: i64 = 1024;
        let to_index = |v: &[i64]| -> Vec<i64> { v.iter().map(|&p| p + TABLE_OFFSET).collect() };
        let frame_idx = Tensor::new(to_index(&frame_index).as_slice(), &self.freqs[0].device())?;
        let height_idx = Tensor::new(to_index(&height_index).as_slice(), &self.freqs[0].device())?;
        let width_idx = Tensor::new(to_index(&width_index).as_slice(), &self.freqs[0].device())?;

        // Gather frequencies for each axis based on mode
        let (frame_freq, height_freq, width_freq) = if self.mode == RopeMode::Complex {
            // Complex mode: gather raw angles, then compute cos/sin at runtime
            let angles = self.angles.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("EmbedNd in Complex mode must have precomputed angles".to_string())
            })?;
            let frame_angles = angles[0].index_select(&frame_idx, 0)?;
            let height_angles = angles[1].index_select(&height_idx, 0)?;
            let width_angles = angles[2].index_select(&width_idx, 0)?;
            let frame_freq = Self::polar(&frame_angles)?;
            let height_freq = Self::polar(&height_angles)?;
            let width_freq = Self::polar(&width_angles)?;
            (frame_freq, height_freq, width_freq)
        } else {
            // Precomputed mode: gather precomputed cos/sin
            let frame_freq = self.freqs[0].index_select(&frame_idx, 0)?;
            let height_freq = self.freqs[1].index_select(&height_idx, 0)?;
            let width_freq = self.freqs[2].index_select(&width_idx, 0)?;
            (frame_freq, height_freq, width_freq)
        };

        // Concatenate: [seq_len, (half0+half1+half2)*2] = [seq_len, hidden_size]
        // where complex is stored as interleaved [real0, imag0, real1, imag1, ...]
        let cat_freq = Tensor::cat(&[&frame_freq, &height_freq, &width_freq], 1)?; // [seq_len, sum(half), 2]

        // Flatten last dim: [seq_len, sum(half), 2] -> [seq_len, sum(half)*2]
        let (s, half_sum, two) = cat_freq.dims3()?;
        Ok(cat_freq.reshape((s, half_sum * two))?)
    }
}

/// Apply rotary embedding to query/key tensors using complex multiplication.
///
/// Args:
///   - `x`: [batch, heads, seq_len, head_dim] where head_dim is even
///   - `freqs_cis`: [seq_len, head_dim] complex as interleaved [real, imag]
/// Returns: [batch, heads, seq_len, head_dim] with RoPE applied
pub fn apply_rope(x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let half = d / 2;

    // x: [b, h, s, d] -> [b, h, s, half, 2] (real, imag)
    let x_reshaped = x.reshape((b, h, s, half, 2))?;
    let x_real = x_reshaped.i((.., .., .., .., 0))?;
    let x_imag = x_reshaped.i((.., .., .., .., 1))?;

    // freqs_cis: [s, d] -> [s, half, 2] (real, imag)
    let freqs_reshaped = freqs_cis.reshape((s, half, 2))?;
    let cos = freqs_reshaped.i((.., .., 0))?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, s, half]
    let sin = freqs_reshaped.i((.., .., 1))?.unsqueeze(0)?.unsqueeze(0)?;

    // Complex multiplication: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
    // Rotate: x * freqs = (x_real + i*x_imag) * (cos + i*sin)
    // = (x_real*cos - x_imag*sin) + i(x_real*sin + x_imag*cos)
    let out_real = x_real.broadcast_mul(&cos)?.broadcast_sub(&x_imag.broadcast_mul(&sin)?)?;
    let out_imag = x_real.broadcast_mul(&sin)?.broadcast_add(&x_imag.broadcast_mul(&cos)?)?;

    // Interleave back: [out_real_0, out_imag_0, out_real_1, out_imag_1, ...]
    let out_real_u = out_real.unsqueeze(4)?;
    let out_imag_u = out_imag.unsqueeze(4)?;
    let out = Tensor::cat(&[&out_real_u, &out_imag_u], 4)?.reshape((b, h, s, d))?;
    Ok(out)
}

/// Compute position indices for 2D image patches (legacy compatibility).
pub fn compute_img_ids(batch: usize, h_patches: usize, w_patches: usize, device: &Device) -> Result<Tensor> {
    let h = Tensor::arange(0, h_patches as i64, device)?.to_dtype(candle_core::DType::F32)?;
    let w = Tensor::arange(0, w_patches as i64, device)?.to_dtype(candle_core::DType::F32)?;
    let grid_h = h.unsqueeze(1)?.expand((h_patches, w_patches))?;
    let grid_w = w.unsqueeze(0)?.expand((h_patches, w_patches))?;
    let ids = Tensor::stack(&[grid_h.flatten_all()?, grid_w.flatten_all()?], 1)?;
    Ok(ids.unsqueeze(0)?.expand((batch as usize, h_patches * w_patches, 2usize))?)
}

/// Compute text position indices (legacy compatibility).
pub fn compute_txt_ids(batch: usize, seq_len: usize, start_idx: usize, device: &Device) -> Result<Tensor> {
    let indices = Tensor::arange(start_idx as i64, (start_idx + seq_len) as i64, device)?
        .to_dtype(candle_core::DType::F32)?;
    let zeros = Tensor::zeros(seq_len, DType::F32, device)?;
    let ids = Tensor::stack(&[indices, zeros], 1)?;
    Ok(ids.unsqueeze(0)?.expand((batch as usize, seq_len as usize, 2usize))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference matching the exact upstream `QwenImage21Rope`
    /// text-to-image formula (direct `coordinate * theta^(-2i/dim)` per axis,
    /// no lookup table): `cos`/`sin` for one (frame, height, width) coordinate.
    fn reference_cos_sin(coord: [i64; 3], axes_dim: [usize; 3], theta: f64) -> (Vec<f32>, Vec<f32>) {
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for (axis, &dim) in axes_dim.iter().enumerate() {
            for i in 0..(dim / 2) {
                let freq = theta.powf(-2.0 * i as f64 / dim as f64);
                let angle = coord[axis] as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        (cos, sin)
    }

    /// Regression test for two real bugs found by cross-checking against a
    /// known-working sibling implementation: (1) the frequency formula computed
    /// `exp(-i/dim)/theta` instead of `theta^(-2i/dim)` — off by orders of
    /// magnitude for every component except i=0 — and (2) the negative-position
    /// table had an off-by-one (`affine(-1.0, -1.0)` instead of `affine(-1.0,
    /// 0.0)`), so every negative height/width coordinate (half the image grid)
    /// read the table entry for `position - 1`. Both silently produced a
    /// near-degenerate RoPE spectrum despite "looking" like it ran correctly.
    #[test]
    fn test_embed_nd_matches_reference_formula() {
        let device = Device::Cpu;
        let axes_dim = [16usize, 56, 56];
        let theta = 10000.0;
        let pe = EmbedNd::new(0, theta, axes_dim, &device).unwrap();

        // text_len=2, image block (frame=1, height=2, width=2): matches the
        // real single-image T2I layout, just small enough to hand-verify.
        let img_shapes = [(1usize, 2usize, 2usize)];
        let image_pad_mask = Tensor::new(&[0u8, 0, 1, 1, 1, 1], &device).unwrap();
        let out = pe.forward(&img_shapes, &image_pad_mask, &device).unwrap();
        assert_eq!(out.dims(), &[6, 128]); // sum(axes_dim) = 16+56+56=128 (half*2 per axis)

        // Text token 0: position 0 on all three axes -> angle 0 everywhere -> cos=1, sin=0.
        let row0: Vec<f32> = out.i(0).unwrap().to_vec1().unwrap();
        assert!(row0.iter().all(|&v| (v.abs() - 1.0).abs() < 1e-4 || v.abs() < 1e-4));

        // Image tokens start at index 2. With height=width=2, h_start=w_start=-1,
        // so the four image tokens are (h,w) in {(-1,-1),(-1,0),(0,-1),(0,0)},
        // each with frame frozen at text_len=2.
        let expected_hw = [(-1i64, -1i64), (-1, 0), (0, -1), (0, 0)];
        for (k, &(h, w)) in expected_hw.iter().enumerate() {
            let row: Vec<f32> = out.i(2 + k).unwrap().to_vec1().unwrap();
            let (cos, sin) = reference_cos_sin([2, h, w], axes_dim, theta);
            let mut expected = Vec::with_capacity(cos.len() * 2);
            for (c, s) in cos.iter().zip(sin.iter()) {
                expected.push(*c);
                expected.push(*s);
            }
            assert_eq!(row.len(), expected.len());
            for (got, want) in row.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-3,
                    "image token {k} (h={h}, w={w}): got {got}, want {want}"
                );
            }
        }
    }
}