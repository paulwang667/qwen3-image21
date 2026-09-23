use candle_core::{Result, Tensor, D};

/// FlowMatchEulerDiscreteScheduler for Qwen-Image-2.1, matching the real
/// `scheduler_config.json`: `use_dynamic_shifting=true`,
/// `time_shift_type="exponential"`, `base_shift=0.5`, `max_shift=0.9`,
/// `base_image_seq_len=256`, `max_image_seq_len=8192`, `shift_terminal=0.02`,
/// `num_train_timesteps=1000`. Without the resolution-dependent shift, higher
/// resolutions (more image tokens) visibly degenerate into high-frequency
/// noise even with many steps — confirmed by A/B testing 256x256 vs 512x512.
#[derive(Debug, Clone)]
pub struct FlowMatchEuler {
    sigmas: Vec<f32>,
    num_inference_steps: usize,
}

const BASE_SHIFT: f32 = 0.5;
const MAX_SHIFT: f32 = 0.9;
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 8192;
const SHIFT_TERMINAL: f32 = 0.02;
const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

impl FlowMatchEuler {
    /// `image_seq_len` is the number of image tokens (patches) being denoised —
    /// used to compute the shift `mu` (real diffusers pipelines pass this from
    /// the packed-latent sequence length, not from height/width directly).
    pub fn new(num_inference_steps: usize, image_seq_len: usize) -> Self {
        let mu = calculate_shift(image_seq_len, BASE_IMAGE_SEQ_LEN, MAX_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_SHIFT);

        let sigma_min = 1.0 / NUM_TRAIN_TIMESTEPS;
        let mut sigmas = linspace(1.0, sigma_min, num_inference_steps);
        for s in sigmas.iter_mut() {
            *s = time_shift_exponential(mu, 1.0, *s);
        }
        stretch_shift_to_terminal(&mut sigmas, SHIFT_TERMINAL);

        Self { sigmas, num_inference_steps }
    }

    /// In flow matching the timestep fed to the model *is* the sigma.
    pub fn timesteps(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Step: euler method for flow matching.
    /// x_t -> x_{t-1}
    pub fn step(&self, model_output: &Tensor, sample: &Tensor, index: usize) -> Result<Tensor> {
        let dt = if index + 1 < self.sigmas.len() {
            self.sigmas[index + 1] - self.sigmas[index]
        } else {
            -self.sigmas[index]
        };

        // Euler step: x_{t-1} = x_t + dt * model_output
        // model_output is the predicted velocity/flow
        let dt_tensor = Tensor::new(&[dt], sample.device())?.to_dtype(sample.dtype())?;
        sample.broadcast_add(&model_output.broadcast_mul(&dt_tensor)?)
    }
}

/// Linear interpolation of the shift `mu` by image sequence length, matching
/// diffusers pipelines' module-level `calculate_shift` helper (not part of the
/// scheduler class itself).
fn calculate_shift(image_seq_len: usize, base_seq_len: usize, max_seq_len: usize, base_shift: f32, max_shift: f32) -> f32 {
    let m = (max_shift - base_shift) / (max_seq_len as f32 - base_seq_len as f32);
    let b = base_shift - m * base_seq_len as f32;
    image_seq_len as f32 * m + b
}

/// `FlowMatchEulerDiscreteScheduler._time_shift_exponential`: exp(mu) / (exp(mu) + (1/t - 1)^shift_exp).
fn time_shift_exponential(mu: f32, shift_exp: f32, t: f32) -> f32 {
    let e_mu = mu.exp();
    e_mu / (e_mu + (1.0 / t - 1.0).powf(shift_exp))
}

/// `FlowMatchEulerDiscreteScheduler.stretch_shift_to_terminal`: rescales so the
/// final sigma equals `shift_terminal` exactly.
fn stretch_shift_to_terminal(sigmas: &mut [f32], shift_terminal: f32) {
    let Some(&last) = sigmas.last() else { return };
    let scale = (1.0 - last) / (1.0 - shift_terminal);
    for s in sigmas.iter_mut() {
        *s = 1.0 - (1.0 - *s) / scale;
    }
}

fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![start];
    }
    (0..n).map(|i| start + (end - start) * (i as f32) / ((n - 1) as f32)).collect()
}

/// Timesteps embedding: sinusoidal positional encoding for timesteps.
pub fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let device = t.device();
    let dt = candle_core::DType::F32;
    let t = t.to_dtype(dt)?;

    let half_dim = dim / 2;
    let exponent: Vec<f32> = (0..half_dim)
        .map(|i| {
            let i = i as f32;
            let log_theta = 10000.0_f32.ln();
            (-log_theta * (i / half_dim as f32)).exp()
        })
        .collect();

    let exp_table = Tensor::new(exponent.as_slice(), device)?;
    // t: [B] -> [B, 1] -> [B, half_dim] -- diffusers uses time_factor=1000.0
    let emb = (t.unsqueeze(1)? * 1000.0)?.broadcast_mul(&exp_table)?;

    // Concatenate sin and cos
    let sin = emb.sin()?;
    let cos = emb.cos()?;

    Tensor::cat(&[sin, cos], D::Minus1)
}

/// TimeEmbedding: two-layer MLP for timestep embedding.
#[derive(Debug, Clone)]
pub struct TimeEmbedding {
    linear_1: candle_nn::Linear,
    linear_2: candle_nn::Linear,
}

impl TimeEmbedding {
    pub fn new(in_dim: usize, hidden_dim: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let linear_1 = candle_nn::linear_b(in_dim, hidden_dim, false, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear_b(hidden_dim, hidden_dim, false, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.linear_1)?.silu()?.apply(&self.linear_2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_match_euler_schedule_endpoints() {
        // t=1.0 always maps to exactly 1.0 under time_shift_exponential (1/t-1 = 0),
        // and stretch_shift_to_terminal is a no-op at that end (it only rescales
        // (1-sigma), which is 0 at the first sigma regardless of scale factor).
        for image_seq_len in [64usize, 1024, 4096] {
            let sched = FlowMatchEuler::new(20, image_seq_len);
            let sigmas = sched.sigmas();
            assert_eq!(sigmas.len(), 20);
            assert!((sigmas[0] - 1.0).abs() < 1e-5, "first sigma should be 1.0, got {}", sigmas[0]);
            assert!(
                (sigmas[sigmas.len() - 1] - SHIFT_TERMINAL).abs() < 1e-4,
                "last sigma should equal shift_terminal ({SHIFT_TERMINAL}), got {}",
                sigmas[sigmas.len() - 1]
            );
            // Monotonically decreasing.
            for w in sigmas.windows(2) {
                assert!(w[0] > w[1], "sigmas should be strictly decreasing: {:?}", sigmas);
            }
        }
    }

    #[test]
    fn test_higher_resolution_shifts_more() {
        // A larger image_seq_len should push mu (and thus the whole shifted
        // schedule) higher — this is the fix for the 512x512 degeneration.
        let low_res = FlowMatchEuler::new(20, 256);
        let high_res = FlowMatchEuler::new(20, 4096);
        // Compare a mid-schedule sigma, where the shift's effect is visible
        // (endpoints are pinned to 1.0 and shift_terminal regardless of mu).
        let mid = 10;
        assert!(
            high_res.sigmas()[mid] > low_res.sigmas()[mid],
            "higher image_seq_len should shift sigmas up: low={} high={}",
            low_res.sigmas()[mid], high_res.sigmas()[mid]
        );
    }
}