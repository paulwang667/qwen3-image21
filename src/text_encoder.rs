//! Text encoder for Qwen-Image-2.1.
//!
//! The real pipeline uses a hybrid Qwen3-Next-style VLM (linear-attention/Mamba2
//! layers mixed with regular attention, plus a non-standard int8 "rotation"
//! quantization scheme) as its text encoder — candle-transformers has no
//! implementation of that architecture or quantization format, and building one
//! from scratch is out of scope here (see CLAUDE.md). Instead, this loads a
//! real, standard dense Qwen3 causal LM (`candle_transformers::models::qwen3::Model`)
//! as a stand-in: it gives the prompt genuine, deterministic, content-dependent
//! influence on generation, unlike the previous placeholder which just returned
//! random noise regardless of prompt text. It will not match the real model's
//! semantics or output quality.
use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, Model};
use tokenizers::Tokenizer;

pub struct TextEncoder {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    /// Number of times to tile the stand-in model's hidden states to reach
    /// the transformer's `joint_attention_dim` (e.g. 4x for 1024 -> 4096).
    tile_factor: usize,
}

impl TextEncoder {
    /// Loads `config.json`, `model.safetensors`, and `tokenizer.json` from
    /// `model_dir` (the standard HuggingFace single-shard layout).
    pub fn load(
        model_dir: impl AsRef<Path>,
        joint_attention_dim: usize,
        device: Device,
    ) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if joint_attention_dim % config.hidden_size != 0 {
            return Err(anyhow!(
                "text encoder hidden_size {} does not evenly divide the transformer's joint_attention_dim {}",
                config.hidden_size, joint_attention_dim
            ));
        }
        let tile_factor = joint_attention_dim / config.hidden_size;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)?
        };
        let model = Model::new(&config, vb)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        Ok(Self { model, tokenizer, device, tile_factor })
    }

    /// Encodes `prompt` into `[1, seq_len, joint_attention_dim]`.
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let hidden = self.model.forward(&input_ids, 0)?; // [1, seq_len, hidden_size]
        let tiles = std::iter::repeat(hidden).take(self.tile_factor).collect::<Vec<_>>();
        Ok(Tensor::cat(&tiles, D::Minus1)?)
    }
}
