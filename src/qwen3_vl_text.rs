//! Text-only subset of Qwen3-VL, adapted from candle-transformers' `qwen3_vl`
//! module (whose `text` submodule is private and whose public `forward()`
//! always projects through `lm_head` and narrows to the last token — built
//! for autoregressive generation, not for extracting per-token hidden states).
//! This is exactly what the official Qwen-Image-2.1 `text_encoder/` weights
//! are: a standard dense Qwen3-VL language model (no vision tower needed,
//! since we only ever encode text prompts, never images).
//!
//! Simplifications versus the vendored original: no vision tower / DeepStack
//! (image-only concerns), no KV-cache (we only ever run one forward pass per
//! prompt, never incremental decoding), and M-RoPE degenerates to plain 1D
//! RoPE for text-only input (all three M-RoPE axes advance identically for
//! text tokens, so this is exact for our case — but would be wrong if this
//! module were ever used with actual image tokens).
use std::sync::Arc;

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
}

/// Matches the real `text_encoder/config.json`'s top-level shape (a nested
/// `text_config`, plus `vision_config`/etc. we don't deserialize at all).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(base: f32, head_dim: usize, max_position_embeddings: usize, device: &Device, dtype: DType) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { cos: freqs.cos()?.to_dtype(dtype)?, sin: freqs.sin()?.to_dtype(dtype)? })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
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
    rotary_emb: Arc<RotaryEmbedding>,
    n_kv_groups: usize,
    softmax_scale: f64,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
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
            rotary_emb,
            n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(xs)?.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;
        let (q, k) = self.rotary_emb.forward(&q, &k)?;

        let q = q.contiguous()?;
        let k = candle_transformers::utils::repeat_kv(k.contiguous()?, self.n_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v.contiguous()?, self.n_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale)?;
        let attn_weights = attn_weights.broadcast_add(causal_mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
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
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, causal_mask)?;
        let xs = (residual + h)?;
        let residual = &xs;
        let h = self.mlp.forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + h
    }
}

/// The official Qwen-Image-2.1 text encoder: a standard dense Qwen3-VL
/// language model, text-only (no vision tower loaded or needed).
pub struct Qwen3VLTextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    device: Device,
    dtype: DType,
}

impl Qwen3VLTextEncoder {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32, cfg.head_dim, cfg.max_position_embeddings, vb.device(), vb_m.dtype(),
        )?);
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(i))?);
        }
        // The final `model.language_model.norm` is deliberately not loaded:
        // Qwen-Image-2.1 conditions on the last decoder state *before* it.
        Ok(Self { embed_tokens, layers, device: vb.device().clone(), dtype: vb.dtype() })
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
        let seq_len = input_ids.dim(1)?;
        let causal_mask = self.causal_mask(seq_len)?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &causal_mask)?;
        }
        Ok(xs)
    }
}
