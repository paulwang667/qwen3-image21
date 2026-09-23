//! Text encoder for Qwen-Image-2.1.
//!
//! Two backends, auto-detected from `config.json`:
//! - The **official** encoder (`text_config` key present, matching Qwen3-VL's
//!   config shape): a real `Qwen3VLTextEncoder` (see qwen3_vl_text.rs) loaded
//!   from the official `Qwen/Qwen-Image-2.1` `text_encoder/` weights. Its
//!   hidden_size (4096) matches the transformer's `joint_attention_dim`
//!   exactly, so no adapter is needed. Its tokenizer lives in a sibling
//!   `processor/` directory (the diffusers pipeline layout splits weights and
//!   tokenizer into separate top-level folders), not alongside the weights.
//! - The **stand-in** encoder (plain `hidden_size` key): the real pipeline's
//!   actual text encoder is a hybrid Qwen3-Next-style VLM (linear-attention/
//!   Mamba2 layers mixed with regular attention, plus a non-standard int8
//!   "rotation" quantization scheme) that candle-transformers doesn't
//!   implement and that's out of scope to build from scratch (see CLAUDE.md).
//!   This loads a standard dense Qwen3 causal LM instead, tiled up to the
//!   transformer's `joint_attention_dim`, as a lower-fidelity fallback when
//!   the official weights aren't available.
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::safetensors_util::resolve_safetensors_paths;

enum Backend {
    Official(crate::qwen3_vl_text::Qwen3VLTextEncoder),
    StandIn { model: candle_transformers::models::qwen3::Model, tile_factor: usize },
}

pub struct TextEncoder {
    backend: Backend,
    tokenizer: Tokenizer,
    device: Device,
}

/// Finds `tokenizer.json` either alongside the weights (the stand-in's
/// self-contained layout) or in a sibling `processor/` directory (the
/// official pipeline's split layout).
fn find_tokenizer(dir: &Path) -> Result<PathBuf> {
    let direct = dir.join("tokenizer.json");
    if direct.exists() {
        return Ok(direct);
    }
    let sibling = dir.parent().map(|p| p.join("processor").join("tokenizer.json"));
    if let Some(p) = &sibling {
        if p.exists() {
            return Ok(p.clone());
        }
    }
    Err(anyhow!(
        "no tokenizer.json found at {} or {:?}/processor/tokenizer.json",
        direct.display(), dir.parent()
    ))
}

impl TextEncoder {
    /// Loads from `model_dir`, auto-detecting the official vs. stand-in layout.
    pub fn load(model_dir: impl AsRef<Path>, joint_attention_dim: usize, device: Device) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let tokenizer_path = find_tokenizer(dir)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        let backend = if config_json.get("text_config").is_some() {
            eprintln!("  Detected official Qwen3-VL text encoder layout");
            let cfg: crate::qwen3_vl_text::Config = serde_json::from_value(config_json)?;
            if cfg.text_config.hidden_size != joint_attention_dim {
                return Err(anyhow!(
                    "official text encoder hidden_size {} does not match the transformer's joint_attention_dim {}",
                    cfg.text_config.hidden_size, joint_attention_dim
                ));
            }
            // The weights may be sharded across multiple files with an index.json;
            // probe every *.safetensors in the dir and resolve shards from whichever exists.
            let first_shard = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
                .ok_or_else(|| anyhow!("no .safetensors file found in {}", dir.display()))?;
            let paths = resolve_safetensors_paths(&first_shard.to_string_lossy())?;
            eprintln!("  Loading official text encoder from {} safetensors shard(s)...", paths.len());
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &device)? };
            Backend::Official(crate::qwen3_vl_text::Qwen3VLTextEncoder::new(&cfg.text_config, vb)?)
        } else {
            eprintln!("  Detected stand-in Qwen3 text encoder layout");
            let config: candle_transformers::models::qwen3::Config = serde_json::from_value(config_json)?;
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
            let model = candle_transformers::models::qwen3::Model::new(&config, vb)?;
            Backend::StandIn { model, tile_factor }
        };

        Ok(Self { backend, tokenizer, device })
    }

    /// Encodes `prompt` into `[1, seq_len, joint_attention_dim]`.
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        let encoding = self.tokenizer.encode(prompt, true).map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        match &mut self.backend {
            Backend::Official(model) => Ok(model.forward(&input_ids)?), // [1, seq_len, hidden_size], already matches joint_attention_dim
            Backend::StandIn { model, tile_factor } => {
                let hidden = model.forward(&input_ids, 0)?; // [1, seq_len, hidden_size]
                if *tile_factor == 1 {
                    return Ok(hidden);
                }
                let tiles = std::iter::repeat(hidden).take(*tile_factor).collect::<Vec<_>>();
                Ok(Tensor::cat(&tiles, D::Minus1)?)
            }
        }
    }
}
