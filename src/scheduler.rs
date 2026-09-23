use candle_core::{Result, Tensor, D};

/// FlowMatchEulerDiscreteScheduler for Qwen-Image.
/// This scheduler implements the Flow Matching Euler method.
#[derive(Debug, Clone)]
pub struct FlowMatchEuler {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    num_inference_steps: usize,
}

impl FlowMatchEuler {
    pub fn new(num_inference_steps: usize) -> Self {
        let sigmas: Vec<f32> = (0..num_inference_steps)
            .map(|i| 1.0 - (i as f32) / (num_inference_steps as f32))
            .collect();
        let timesteps: Vec<f32> = sigmas.iter().map(|&s| s).collect();

        Self {
            sigmas,
            timesteps,
            num_inference_steps,
        }
    }

    /// Set custom sigmas.
    pub fn set_sigmas(&mut self, sigmas: Vec<f32>) {
        self.sigmas = sigmas.clone();
        self.timesteps = sigmas;
        self.num_inference_steps = self.sigmas.len();
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Step: euler method for flow matching.
    /// x_t -> x_{t-1}
    pub fn step(&self, model_output: &Tensor, _timestep: f32, sample: &Tensor, index: usize) -> Result<Tensor> {
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

    /// Compute shift for image sequence length.
    /// Used to adjust sigma schedule based on image resolution.
    pub fn calculate_shift(image_seq_len: usize, base_seq_len: usize, max_seq_len: usize, base_shift: f32, max_shift: f32) -> f32 {
        let m = (max_shift - base_shift) / (max_seq_len as f32 - base_seq_len as f32);
        let b = base_shift - m * base_seq_len as f32;
        let mu = image_seq_len as f32 * m + b;
        mu
    }
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