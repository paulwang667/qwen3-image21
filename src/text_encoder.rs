//! Text encoder for Qwen-Image-2.1.
//!
//! Two backends, auto-detected from `config.json`:
//! - The **official** encoder (`text_config` key present): the Qwen3-VL model
//!   from the official `Qwen/Qwen-Image-2.1` `text_encoder/` weights — the
//!   language model (qwen3_vl_text.rs, hidden size 4096 = the transformer's
//!   `joint_attention_dim`) plus its vision tower (qwen3_vl_vision.rs), which
//!   reads condition images for image-conditioned generation. Its tokenizer
//!   lives in the sibling `processor/` directory of the diffusers layout.
//! - The **stand-in** encoder (plain `hidden_size` key): a standard dense Qwen3
//!   causal LM whose hidden states are tiled up to `joint_attention_dim` — a
//!   low-fidelity, text-only fallback for when the official weights don't fit.
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::safetensors_util::resolve_safetensors_paths;

/// System prompt of the upstream Qwen-Image-2.1 text-to-image prompt template.
const OFFICIAL_SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// The official Qwen3-VL encoder: language model plus the vision tower that
/// reads condition images.
struct Official {
    text: crate::qwen3_vl_text::Qwen3VLTextEncoder,
    vision: crate::qwen3_vl_vision::Qwen3VLVisionModel,
    image_token_id: u32,
    spatial_merge_size: usize,
}

enum Backend {
    Official(Official),
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
    /// Loads from `model_dir`, auto-detecting the official vs. stand-in layout,
    /// with weights in `dtype` on `device` (e.g. F32 on the CPU to keep the
    /// ~34 GB F32 encoder off the GPU, or BF16 on the GPU to halve it).
    pub fn load(model_dir: impl AsRef<Path>, joint_attention_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        Self::load_with(model_dir.as_ref(), joint_attention_dim, device, dtype, false)
    }

    /// Like [`Self::load`], but the official encoder's language-model layers are
    /// streamed onto `device` one at a time during encoding instead of being
    /// resident (see `qwen3_vl_text::Qwen3VLTextEncoder::new_streamed`). The
    /// stand-in layout is always loaded resident.
    pub fn load_streamed(model_dir: impl AsRef<Path>, joint_attention_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        Self::load_with(model_dir.as_ref(), joint_attention_dim, device, dtype, true)
    }

    fn load_with(dir: &Path, joint_attention_dim: usize, device: Device, dtype: DType, streamed: bool) -> Result<Self> {
        let config_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let tokenizer_path = find_tokenizer(dir)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        let backend = if config_json.get("text_config").is_some() {
            eprintln!("  Detected official Qwen3-VL text encoder layout");
            let cfg: crate::qwen3_vl_text::Config = serde_json::from_value(config_json.clone())?;
            let vision_cfg: crate::qwen3_vl_vision::VisionConfig = serde_json::from_value(config_json["vision_config"].clone())?;
            let image_token_id = config_json["image_token_id"]
                .as_u64()
                .ok_or_else(|| anyhow!("text encoder config.json has no image_token_id"))? as u32;
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
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)? };
            Backend::Official(Official {
                text: if streamed {
                    crate::qwen3_vl_text::Qwen3VLTextEncoder::new_streamed(&cfg.text_config, vb.clone())?
                } else {
                    crate::qwen3_vl_text::Qwen3VLTextEncoder::new(&cfg.text_config, vb.clone())?
                },
                vision: crate::qwen3_vl_vision::Qwen3VLVisionModel::new(&vision_cfg, vb.pp("model").pp("visual"))?,
                image_token_id,
                spatial_merge_size: vision_cfg.spatial_merge_size,
            })
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
                VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, &device)?
            };
            let model = candle_transformers::models::qwen3::Model::new(&config, vb)?;
            Backend::StandIn { model, tile_factor }
        };

        Ok(Self { backend, tokenizer, device })
    }

    /// Encodes `prompt` into `[1, seq_len, joint_attention_dim]`.
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        let tokenizer = &self.tokenizer;
        let tokenize = |text: &str| -> Result<Vec<u32>> {
            Ok(tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer encode failed: {e}"))?.get_ids().to_vec())
        };
        match &mut self.backend {
            Backend::Official(Official { text: model, .. }) => {
                // Upstream wraps the prompt in this raw template (not
                // apply_chat_template), encodes it, then drops the hidden states
                // of the system-role prefix.
                let prompt = if prompt.is_empty() { " " } else { prompt };
                let system = format!("<|im_start|>system\n{OFFICIAL_SYSTEM_PROMPT}<|im_end|>\n");
                let ids = tokenize(&format!("{system}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"))?;
                let drop_idx = tokenize(&system)?.len();
                let input_ids = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
                let hidden = model.forward(&input_ids)?; // [1, seq_len, 4096]
                Ok(hidden.narrow(1, drop_idx, ids.len() - drop_idx)?)
            }
            Backend::StandIn { model, tile_factor } => {
                let input_ids = Tensor::new(tokenize(prompt)?.as_slice(), &self.device)?.unsqueeze(0)?;
                let hidden = model.forward(&input_ids, 0)?; // [1, seq_len, hidden_size]
                if *tile_factor == 1 {
                    return Ok(hidden);
                }
                let tiles = std::iter::repeat(hidden).take(*tile_factor).collect::<Vec<_>>();
                Ok(Tensor::cat(&tiles, D::Minus1)?)
            }
        }
    }

    /// Encodes `prompt` with condition images (upstream `QwenImage21Pipeline`'s
    /// image-conditioned template). Returns the prompt embeddings
    /// `[1, seq_len, 4096]` and, per returned token, whether it is an
    /// `<|image_pad|>` slot (where the transformer substitutes the condition
    /// image's latents). Both exclude the dropped system prefix.
    pub fn encode_with_images(&mut self, prompt: &str, images: &[crate::condition_image::ConditionImage]) -> Result<(Tensor, Vec<bool>)> {
        let Backend::Official(enc) = &mut self.backend else {
            return Err(anyhow!("condition images need the official Qwen3-VL text encoder"));
        };
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let system = format!("<|im_start|>system\n{OFFICIAL_SYSTEM_PROMPT}<|im_end|>\n");
        // One `<|image_pad|>` per merged vision patch, as the Qwen3-VL processor expands it.
        let mut vision_block = String::new();
        for (i, img) in images.iter().enumerate() {
            if i > 0 {
                vision_block.push(' ');
            }
            vision_block.push_str(&format!("<image{}><|vision_start|>{}<|vision_end|>", i + 1, "<|image_pad|>".repeat(img.num_image_tokens())));
        }
        let text = format!("{system}<|im_start|>user\n{vision_block}{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let tokenize = |t: &str| -> Result<Vec<u32>> {
            Ok(self.tokenizer.encode(t, true).map_err(|e| anyhow!("tokenizer encode failed: {e}"))?.get_ids().to_vec())
        };
        let ids = tokenize(&text)?;
        let drop_idx = tokenize(&system)?.len();

        let features = if images.is_empty() {
            None
        } else {
            // The images may have been prepared on another device than the encoder's.
            let pixel_values = Tensor::cat(&images.iter().map(|i| &i.pixel_values).collect::<Vec<_>>(), 0)?.to_device(&self.device)?;
            let grid = Tensor::cat(&images.iter().map(|i| &i.grid_thw).collect::<Vec<_>>(), 0)?.to_device(&self.device)?;
            let (embeds, deepstack) = enc.vision.forward(&pixel_values, &grid)?;
            let m = enc.spatial_merge_size;
            let merged_grids = grid.to_vec2::<u32>()?.iter().map(|g| (g[0] as usize, g[1] as usize / m, g[2] as usize / m)).collect();
            Some(crate::qwen3_vl_text::ImageFeatures { embeds, deepstack, merged_grids, image_token_id: enc.image_token_id })
        };
        let input_ids = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden = enc.text.forward_with_images(&input_ids, features.as_ref())?;
        let slots = ids[drop_idx..].iter().map(|&id| id == enc.image_token_id).collect();
        Ok((hidden.narrow(1, drop_idx, ids.len() - drop_idx)?, slots))
    }

}
