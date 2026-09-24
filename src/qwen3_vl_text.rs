//! Qwen3-VL language model, adapted from candle-transformers' `qwen3_vl`
//! module (whose `text` submodule is private and whose public `forward()`
//! always projects through `lm_head` and narrows to the last token — built
//! for autoregressive generation, not for extracting per-token hidden states).
//! This is the official Qwen-Image-2.1 `text_encoder/` language model.
//!
//! Versus the vendored original: no KV-cache (one forward pass per prompt), and
//! it implements what candle's version leaves out for image inputs — the 3D
//! multimodal RoPE positions of upstream `get_rope_index` with the checkpoint's
//! interleaved `mrope_section`. Image features (from `qwen3_vl_vision`) replace
//! the `<|image_pad|>` embeddings, and DeepStack features are added to those
//! positions after the first layers. For text-only input every token's three
//! position components are equal, so this reduces to plain 1D RoPE.
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Activation, Embedding, RmsNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// Frequencies per (T, H, W) axis, interleaved as T,H,W,T,H,W,... (the
    /// checkpoint sets `mrope_interleaved: true`).
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

/// Matches the real `text_encoder/config.json`'s top-level shape (a nested
/// `text_config`, plus `vision_config`/etc. we don't deserialize at all).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

/// Interleaved multimodal RoPE (upstream `Qwen3VLTextRotaryEmbedding` with
/// `apply_interleaved_mrope`): frequency `j` takes its position from the H axis
/// when `j % 3 == 1 && j < 3 * section[1]`, from W when `j % 3 == 2 && j < 3 *
/// section[2]`, and from T otherwise.
#[derive(Debug, Clone)]
struct MultimodalRope {
    inv_freq: Vec<f32>,
    section: [usize; 3],
}

impl MultimodalRope {
    fn new(base: f32, head_dim: usize, section: [usize; 3]) -> Self {
        let inv_freq = (0..head_dim).step_by(2).map(|i| 1f32 / base.powf(i as f32 / head_dim as f32)).collect();
        Self { inv_freq, section }
    }

    /// `cos`/`sin` of shape `[seq_len, head_dim / 2]` for per-token `(t, h, w)` positions.
    fn cos_sin(&self, positions: &[[i64; 3]], device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for p in positions {
            for (j, f) in self.inv_freq.iter().enumerate() {
                let axis = if j % 3 == 1 && j < 3 * self.section[1] {
                    1
                } else if j % 3 == 2 && j < 3 * self.section[2] {
                    2
                } else {
                    0
                };
                let angle = p[axis] as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let shape = (positions.len(), half);
        Ok((Tensor::from_vec(cos, shape, device)?.to_dtype(dtype)?, Tensor::from_vec(sin, shape, device)?.to_dtype(dtype)?))
    }
}

/// Upstream `get_rope_index` for one unpadded sequence. Text tokens advance all
/// three axes together; a run of image tokens with merged grid `(t, h, w)` gets
/// `(start + t_i, start + h_i, start + w_i)`, after which positions resume at
/// `start + max(h, w)`.
fn mrope_positions(input_ids: &[u32], image_token_id: u32, grids: &[(usize, usize, usize)]) -> Result<Vec<[i64; 3]>> {
    let mut out = Vec::with_capacity(input_ids.len());
    let mut grids = grids.iter();
    let (mut pos, mut i) = (0i64, 0usize);
    while i < input_ids.len() {
        if input_ids[i] != image_token_id {
            out.push([pos; 3]);
            pos += 1;
            i += 1;
            continue;
        }
        let Some(&(t, h, w)) = grids.next() else {
            candle_core::bail!("more <|image_pad|> runs than image grids");
        };
        let run = input_ids[i..].iter().take_while(|&&id| id == image_token_id).count();
        if run != t * h * w {
            candle_core::bail!("image token run of {run} does not match grid {t}x{h}x{w}");
        }
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    out.push([pos + ti as i64, pos + hi as i64, pos + wi as i64]);
                }
            }
        }
        pos += h.max(w) as i64;
        i += run;
    }
    Ok(out)
}

/// Vision-tower output for the `<|image_pad|>` tokens of one prompt, in order.
pub struct ImageFeatures {
    /// `[num_image_tokens, hidden_size]` merged patch embeddings.
    pub embeds: Tensor,
    /// One `[num_image_tokens, hidden_size]` tensor per DeepStack layer.
    pub deepstack: Vec<Tensor>,
    /// Merged `(t, h, w)` grid of each image (vision grid / spatial merge size).
    pub merged_grids: Vec<(usize, usize, usize)>,
    pub image_token_id: u32,
}

/// Returns `base` with the rows at `mask` positions replaced by (or, with
/// `add`, increased by) consecutive rows of `rows`. `base` is `[1, L, D]`.
fn scatter_rows(base: &Tensor, mask: &[bool], rows: &Tensor, add: bool) -> Result<Tensor> {
    let mut pieces = Vec::new();
    let (mut i, mut r) = (0usize, 0usize);
    while i < mask.len() {
        let run = mask[i..].iter().take_while(|&&m| m == mask[i]).count();
        let seg = base.narrow(1, i, run)?;
        pieces.push(if mask[i] {
            let img = rows.narrow(0, r, run)?.unsqueeze(0)?.to_dtype(base.dtype())?;
            r += run;
            if add { (seg + img)? } else { img }
        } else {
            seg
        });
        i += run;
    }
    Tensor::cat(&pieces, 1)
}

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_kv_groups: usize,
    softmax_scale: f64,
}

impl Attention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(xs)?.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;

        let q = q.contiguous()?;
        let k = candle_transformers::utils::repeat_kv(k.contiguous()?, self.n_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v.contiguous()?, self.n_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale)?;
        let attn_weights = attn_weights.broadcast_add(causal_mask)?;
        // Softmax in F32 even for BF16/F16 weights.
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        let attn_out = attn_weights.matmul(&v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_out)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, causal_mask, cos, sin)?;
        let xs = (residual + h)?;
        let residual = &xs;
        let h = self.mlp.forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + h
    }
}

/// Where the decoder layers' weights live.
enum Layers {
    /// All layers on the device.
    Resident(Vec<DecoderLayer>),
    /// Each layer is built from the (mmap-backed) `vb` right before it runs and
    /// dropped right after, so only one layer's weights (~0.4 GB in BF16) are
    /// on the device at a time; the embedding table stays in host memory.
    Streamed { cfg: TextConfig, vb: VarBuilder<'static> },
}

/// The official Qwen-Image-2.1 text encoder: the Qwen3-VL language model.
pub struct Qwen3VLTextEncoder {
    embed_tokens: Embedding,
    layers: Layers,
    rope: MultimodalRope,
    device: Device,
    dtype: DType,
}

impl Qwen3VLTextEncoder {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_l.pp(i))?);
        }
        // The final `model.language_model.norm` is deliberately not loaded:
        // Qwen-Image-2.1 conditions on the last decoder state *before* it.
        Ok(Self { embed_tokens, layers: Layers::Resident(layers), rope: Self::rope(cfg), device: vb.device().clone(), dtype: vb.dtype() })
    }

    /// Like [`Self::new`], but streams the decoder layers onto `vb`'s device
    /// one at a time during each forward pass (see [`Layers::Streamed`]).
    pub fn new_streamed(cfg: &TextConfig, vb: VarBuilder<'static>) -> Result<Self> {
        let (device, dtype) = (vb.device().clone(), vb.dtype());
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens").set_device(Device::Cpu))?;
        let layers = Layers::Streamed { cfg: cfg.clone(), vb: vb_m.pp("layers") };
        Ok(Self { embed_tokens, layers, rope: Self::rope(cfg), device, dtype })
    }

    fn rope(cfg: &TextConfig) -> MultimodalRope {
        let section = match cfg.rope_scaling.as_ref().map(|r| r.mrope_section.as_slice()) {
            Some(&[t, h, w]) => [t, h, w],
            // No section: every frequency reads the T axis, i.e. plain 1D RoPE.
            _ => [cfg.head_dim / 2, 0, 0],
        };
        MultimodalRope::new(cfg.rope_theta as f32, cfg.head_dim, section)
    }

    fn causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), &self.device)?.to_dtype(self.dtype)
    }

    /// Encodes token ids `[1, seq_len]` into the final decoder layer's output
    /// `[1, seq_len, hidden_size]`, before the model's final RMSNorm.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_images(input_ids, None)
    }

    /// Like [`Self::forward`], with image features for the prompt's
    /// `<|image_pad|>` tokens (upstream `Qwen3VLModel.forward` for one
    /// unpadded sequence).
    pub fn forward_with_images(&self, input_ids: &Tensor, images: Option<&ImageFeatures>) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let ids_on_table = input_ids.to_device(self.embed_tokens.embeddings().device())?;
        let mut xs = self.embed_tokens.forward(&ids_on_table)?.to_device(&self.device)?;
        let (positions, image_mask) = match images {
            Some(img) => {
                let mask: Vec<bool> = ids.iter().map(|&id| id == img.image_token_id).collect();
                xs = scatter_rows(&xs, &mask, &img.embeds, false)?;
                (mrope_positions(&ids, img.image_token_id, &img.merged_grids)?, Some(mask))
            }
            None => ((0..seq_len as i64).map(|p| [p; 3]).collect(), None),
        };
        let (cos, sin) = self.rope.cos_sin(&positions, &self.device, self.dtype)?;
        let causal_mask = self.causal_mask(seq_len)?;
        let num_layers = match &self.layers {
            Layers::Resident(layers) => layers.len(),
            Layers::Streamed { cfg, .. } => cfg.num_hidden_layers,
        };
        for i in 0..num_layers {
            xs = match &self.layers {
                Layers::Resident(layers) => layers[i].forward(&xs, &causal_mask, &cos, &sin)?,
                Layers::Streamed { cfg, vb } => DecoderLayer::new(cfg, vb.pp(i))?.forward(&xs, &causal_mask, &cos, &sin)?,
            };
            if let (Some(img), Some(mask)) = (images, &image_mask) {
                if let Some(feat) = img.deepstack.get(i) {
                    xs = scatter_rows(&xs, mask, feat, true)?;
                }
            }
        }
        Ok(xs)
    }
}
