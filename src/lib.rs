pub mod gguf_loader;
pub mod gguf_mapping;
pub mod gguf_mapped;
pub mod rope;
pub mod attention_mask;
pub mod scheduler;
pub mod transformer;
pub mod quantized_transformer;
pub mod vae;
pub mod pipeline;
pub mod text_encoder;
pub mod qwen3_vl_text;
pub mod qwen3_vl_vision;
pub mod condition_image;
pub mod joint_layout;
pub mod safetensors_util;
pub mod model_dir;

/// Inference precision for model weights and compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Precision {
    /// 32-bit float (default, most accurate)
    F32,
    /// 16-bit float (half the memory, near-identical quality)
    F16,
    /// Brain float 16 (same memory as f16, larger range, less precision)
    BF16,
}

impl Precision {
    pub fn as_dtype(&self) -> candle_core::DType {
        match self {
            Self::F32 => candle_core::DType::F32,
            Self::F16 => candle_core::DType::F16,
            Self::BF16 => candle_core::DType::BF16,
        }
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::F32
    }
}
