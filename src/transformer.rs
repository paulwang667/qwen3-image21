use candle_core::{Module, Result, Tensor, D, DType, Device};
use candle_nn::{LayerNorm, Linear, RmsNorm};
use super::attention_mask::{build_block_causal_mask, build_token_metadata};
use super::rope::{apply_rope, EmbedNd};
use super::scheduler::TimeEmbedding;

/// Key-Value cache for transformer blocks.
/// Stores cached key/value tensors across denoising steps to avoid recomputation.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached key tensor: [batch, heads, seq_len, dim_head]
    pub k_cache: Option<Tensor>,
    /// Cached value tensor: [batch, heads, seq_len, dim_head]
    pub v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k_cache: None, v_cache: None }
    }

    /// Update cache with new key/value tensors.
    /// Returns the concatenated cache for attention computation.
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k_cat, v_cat) = if let (Some(kc), Some(vc)) = (&self.k_cache, &self.v_cache) {
            let k_new = Tensor::cat(&[kc, new_k], 2)?;
            let v_new = Tensor::cat(&[vc, new_v], 2)?;
            (k_new, v_new)
        } else {
            (new_k.clone(), new_v.clone())
        };
        self.k_cache = Some(k_cat.clone());
        self.v_cache = Some(v_cat.clone());
        Ok((k_cat, v_cat))
    }

    /// Get cached k/v tensors (for prefill pass).
    pub fn get_cached(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.k_cache, &self.v_cache) {
            (Some(kc), Some(vc)) => Some((kc, vc)),
            _ => None,
        }
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.k_cache.is_none()
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Qwen-Image-2.1 Transformer config (32 Single-Stream DiT layers).
#[derive(Debug, Clone)]
pub struct Config {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub pooled_projection_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f64,
    pub use_additional_t_cond: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            patch_size: 1,
            in_channels: 64,
            out_channels: 64,
            num_layers: 32,
            num_attention_heads: 32,
            attention_head_dim: 128,
            joint_attention_dim: 4096,
            pooled_projection_dim: 4096,
            axes_dims_rope: [16, 56, 56],
            rope_theta: 10000.0,
            use_additional_t_cond: false,
        }
    }
}

impl Config {
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    
    pub fn mlp_hidden_size(&self) -> usize {
        3 * self.hidden_size()
    }
}

/// Timestep projection embeddings for Qwen-Image.
#[derive(Debug, Clone)]
pub struct QwenTimestepProj {
    timestep_embedder: TimeEmbedding,
}

impl QwenTimestepProj {
    fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        // Caller already scopes vb to "time_text_embed.timestep_embedder" — don't
        // apply the "timestep_embedder" prefix a second time here.
        let timestep_embedder = TimeEmbedding::new(
            256, // t_dim
            cfg.hidden_size(),
            vb,
        )?;
        
        Ok(Self {
            timestep_embedder,
        })
    }
    
    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        self.timestep_embedder.forward(timestep)
    }
}

/// Single-stream Attention module for Qwen-Image-2.1.
#[derive(Debug, Clone)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
    dim_head: usize,
    inner_dim: usize,
    dropout: f64,
}

impl Attention {
    fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size();
        let d_head = cfg.attention_head_dim;
        
        let to_q = candle_nn::linear_b(cfg.hidden_size(), h_sz, false, vb.pp("to_q"))?;
        let to_k = candle_nn::linear_b(cfg.hidden_size(), h_sz, false, vb.pp("to_k"))?;
        let to_v = candle_nn::linear_b(cfg.hidden_size(), h_sz, false, vb.pp("to_v"))?;
        let to_out = candle_nn::linear_b(h_sz, cfg.hidden_size(), false, vb.pp("to_out.0"))?;
        
        let norm_q_weight = vb.pp("norm_q").get(d_head, "weight")?;
        let norm_q = RmsNorm::new(norm_q_weight, 1e-6);
        
        let norm_k_weight = vb.pp("norm_k").get(d_head, "weight")?;
        let norm_k = RmsNorm::new(norm_k_weight, 1e-6);
        
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            norm_q,
            norm_k,
            heads: cfg.num_attention_heads,
            dim_head: d_head,
            inner_dim: h_sz,
            dropout: 0.0,
        })
    }

    fn forward(&self, x: &Tensor, pe: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        
        // Project
        let q = x.apply(&self.to_q)?;
        let k = x.apply(&self.to_k)?;
        let v = x.apply(&self.to_v)?;
        
        // Reshape to [B, H, S, D]
        let q = q.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?;
        let k = k.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?;
        let v = v.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?.contiguous()?; // Metal matmul needs contiguous operands
        
        // Apply Q/K normalization
        let q = q.apply(&self.norm_q)?;
        let k = k.apply(&self.norm_k)?;
        
        // Apply RoPE
        let q = apply_rope(&q, pe)?;
        let k = apply_rope(&k, pe)?;
        // Attention
        let scale = (self.dim_head as f64).powf(-0.5);
        let mut attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            // mask: [batch, 1, seq_q, seq_kv] bool, True = attend
            // Convert to additive mask: 0 for attend, -inf for masked
            let neg_inf = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?.broadcast_as(mask.dims())?.contiguous()?;
            let zero = Tensor::new(0.0f32, attn_weights.device())?.broadcast_as(mask.dims())?.contiguous()?;
            let additive_mask = mask.where_cond(&zero, &neg_inf)?; // [batch, 1, seq, seq]
            attn_weights = attn_weights.broadcast_add(&additive_mask)?;
        }
        
        let mut attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        
        // Apply dropout if enabled
        if self.dropout > 0.0 {
            let mask = attn_weights.rand_like(0.0, 1.0)?.gt(self.dropout as f32)?;
            attn_weights = attn_weights.mul(&mask)?.affine(1.0 / (1.0 - self.dropout), 0.0)?;
        }
        
        let attn_out = attn_weights.matmul(&v)?;
        
        // Reshape back: [B, H, S, D] -> [B, S, D]
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq, self.inner_dim))?;
        
        // Output projection
        attn_out.apply(&self.to_out)
    }
    
    /// Cached forward pass with KV cache support.
    /// Uses pre-computed key/value tensors from previous steps to avoid recomputation.
    fn forward_cached(
        &self,
        x: &Tensor,
        pe: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        
        // Project query
        let q = x.apply(&self.to_q)?;
        let k = x.apply(&self.to_k)?;
        let v = x.apply(&self.to_v)?;
        
        // Reshape to [B, H, S, D]
        let q = q.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?;
        let k = k.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?;
        let v = v.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?.contiguous()?; // Metal matmul needs contiguous operands
        
        // Apply Q/K normalization
        let q = q.apply(&self.norm_q)?;
        let k = k.apply(&self.norm_k)?;
        
        // Apply RoPE
        let q = apply_rope(&q, pe)?;
        let k = apply_rope(&k, pe)?;
        
        // Concatenate with cached k/v if available
        let (k_full, v_full) = if let Some(cache) = kv_cache.as_ref() {
            if let Some((kc, vc)) = cache.get_cached() {
                let k_cat = Tensor::cat(&[kc, &k], 2)?;
                let v_cat = Tensor::cat(&[vc, &v], 2)?;
                (k_cat, v_cat)
            } else {
                (k, v)
            }
        } else {
            (k, v)
        };
        
        // Update cache with new k/v
        if let Some(cache) = kv_cache {
            let _ = cache.update(&k_full, &v_full);
        }
        
        // Attention
        let scale = (self.dim_head as f64).powf(-0.5);
        let mut attn_weights = q.matmul(&k_full.transpose(2, 3)?)?.affine(scale, 0.0)?;
        
        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            let neg_inf = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?.broadcast_as(mask.dims())?.contiguous()?;
            let zero = Tensor::new(0.0f32, attn_weights.device())?.broadcast_as(mask.dims())?.contiguous()?;
            let additive_mask = mask.where_cond(&zero, &neg_inf)?;
            attn_weights = attn_weights.broadcast_add(&additive_mask)?;
        }
        
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_out = attn_weights.matmul(&v_full)?;
        
        // Reshape back: [B, H, S, D] -> [B, S, D]
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq, self.inner_dim))?;
        
        // Output projection
        attn_out.apply(&self.to_out)
    }

}

/// Gated MLP (SwiGLU) for Qwen-Image-2.1.
/// 
/// Architecture:
/// - proj: [hidden_size] -> [mlp_hidden_size]
/// - gate_layer: [hidden_size] -> [mlp_hidden_size]
/// - out: [mlp_hidden_size] -> [hidden_size]
/// 
/// Forward: out(silu(gate_layer(x)) * proj(x)) — matches upstream
/// `QwenImage21SwiGLUFeedForward`; the activation is on `gate_layer`, not `proj`.
#[derive(Debug, Clone)]
struct GatedMlp {
    proj: Linear,
    gate_layer: Linear,
    out: Linear,
}

impl GatedMlp {
    fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size();
        let mlp_hidden = cfg.mlp_hidden_size();
        
        let proj = candle_nn::linear_b(hidden, mlp_hidden, false, vb.pp("proj"))?;
        let gate_layer = candle_nn::linear_b(hidden, mlp_hidden, false, vb.pp("gate_layer"))?;
        let out = candle_nn::linear_b(mlp_hidden, hidden, false, vb.pp("out"))?;
        
        Ok(Self { proj, gate_layer, out })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let proj = x.apply(&self.proj)?;
        let gate = x.apply(&self.gate_layer)?;
        let gated = gate.silu()?.broadcast_mul(&proj)?;
        gated.apply(&self.out)
    }
}

impl Module for GatedMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

/// Global Modulation layer for Qwen-Image-2.1.
/// Produces 4 outputs: [scale1, gate1, scale2, gate2]
/// Shared across all 32 transformer blocks.
#[derive(Debug, Clone)]
struct Modulation {
    lin: Linear,
}

impl Modulation {
    fn new(dim: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        // PyTorch Sequential: SiLU(0) -> Linear(1)
        let lin = candle_nn::linear_b(dim, 4 * dim, false, vb.pp("1"))?;
        Ok(Self { lin })
    }
    
    fn forward(&self, t_emb: &Tensor) -> Result<Tensor> {
        t_emb.silu()?.apply(&self.lin)
    }
}

/// Create a fixed LayerNorm (ones weight, no bias, no learnable params).
/// Qwen-Image-2.1 uses elementwise_affine=False LayerNorm in blocks and final layer.
fn fixed_layer_norm(dim: usize, device: &Device) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, DType::F32, device)?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

/// Single-stream Transformer block for Qwen-Image-2.1.
/// Modulation is shared (passed in from model-level global Modulation).
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: GatedMlp,
}

impl TransformerBlock {
    fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size();
        let device = vb.device();

        let norm1 = fixed_layer_norm(hidden, device)?;
        let attn = Attention::new(cfg, vb.pp("attn"))?;
        let norm2 = fixed_layer_norm(hidden, device)?;
        let mlp = GatedMlp::new(cfg, vb.pp("img_mlp"))?;

        Ok(Self { norm1, attn, norm2, mlp })
    }

    fn forward(
        &self,
        x: &Tensor,
        modulation: &Tensor,
        pe: &Tensor,
        attention_mask: Option<&Tensor>,
        _target_token_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // modulation: [B, seq, 4*hidden] -> 4 chunks of [B, seq, hidden]. Already
        // per-token (text tokens carry a frozen t=0 row, image tokens the real
        // timestep's row — see QwenImageTransformer::forward's causal_condition
        // construction), so no broadcast unsqueeze is needed here.
        let parts = modulation.chunk(4, D::Minus1)?;
        let (scale1, gate1, scale2, gate2) = (&parts[0], &parts[1], &parts[2], &parts[3]);

        // norm1(x) * (1 + scale1), gate1
        let normed = self.norm1.forward(x)?;
        let scale1_u = scale1.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale1_u)?;
        let attn_out = self.attn.forward(&normed, pe, attention_mask)?;
        let gate1_u = gate1.tanh()?;
        let x = x.broadcast_add(&gate1_u.broadcast_mul(&attn_out)?)?;

        // norm2(x) * (1 + scale2), gate2
        let normed = self.norm2.forward(&x)?;
        let scale2_u = scale2.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale2_u)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let gate2_u = gate2.tanh()?;
        let x = x.broadcast_add(&gate2_u.broadcast_mul(&mlp_out)?)?;

        Ok(x)
    }

    /// Cached forward pass with KV cache support.
    fn forward_cached(
        &self,
        x: &Tensor,
        modulation: &Tensor,
        pe: &Tensor,
        attention_mask: Option<&Tensor>,
        _target_token_mask: Option<&Tensor>,
        kv_caches: &mut [KVCache],
    ) -> Result<Tensor> {
        // See forward() above: modulation is already per-token [B, seq, 4*hidden].
        let parts = modulation.chunk(4, D::Minus1)?;
        let (scale1, gate1, scale2, gate2) = (&parts[0], &parts[1], &parts[2], &parts[3]);

        // norm1(x) * (1 + scale1), gate1
        let normed = self.norm1.forward(x)?;
        let scale1_u = scale1.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale1_u)?;
        let attn_out = if let Some(cache) = kv_caches.get_mut(0) {
            self.attn.forward_cached(&normed, pe, attention_mask, Some(cache))?
        } else {
            self.attn.forward_cached(&normed, pe, attention_mask, None)?
        };
        let gate1_u = gate1.tanh()?;
        let x = x.broadcast_add(&gate1_u.broadcast_mul(&attn_out)?)?;

        // norm2(x) * (1 + scale2), gate2
        let normed = self.norm2.forward(&x)?;
        let scale2_u = scale2.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale2_u)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let gate2_u = gate2.tanh()?;
        let x = x.broadcast_add(&gate2_u.broadcast_mul(&mlp_out)?)?;

        Ok(x)
    }
}

/// Final layer for Qwen-Image-2.1 (AdaLayerNormContinuous, scale-only).
/// norm_out: norm(x) * (1 + linear(silu(temb)))
/// proj_out is separate at the top level of the transformer.
#[derive(Debug, Clone)]
struct FinalLayer {
    norm: LayerNorm,
    linear: Linear,
}

impl FinalLayer {
    fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size();
        let device = vb.device();
        let norm = fixed_layer_norm(hidden, device)?;
        let linear = candle_nn::linear_b(hidden, hidden, false, vb.pp("linear"))?;
        Ok(Self { norm, linear })
    }

    fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
        // AdaLayerNormContinuous: norm(x) * (1 + linear(silu(t_emb)))
        let scale = t_emb.silu()?.apply(&self.linear)?;
        let scale_u = scale.unsqueeze(1)?.affine(1.0, 1.0)?;  // [B, 1, hidden]
        let normed = self.norm.forward(x)?;
        normed.broadcast_mul(&scale_u)
    }
}

/// Qwen-Image-2.1 Transformer (32 Single-Stream DiT layers).
#[derive(Debug, Clone)]
pub struct QwenImageTransformer {
    img_in: Linear,
    txt_in: Option<TextEmbedder>,
    time_text_embed: QwenTimestepProj,
    modulation: Modulation,
    pub transformer_blocks: Vec<TransformerBlock>,
    final_layer: FinalLayer,
    proj_out: Linear,
    cfg: Config,
    device: Device,
}

/// Text embedding layer (QwenImage21TextProjection).
/// Order: text_norm(ZeroCenterRMSNorm) -> in_layer -> GELU(tanh) -> out_layer
#[derive(Debug, Clone)]
struct TextEmbedder {
    text_norm: RmsNorm,
    in_layer: Linear,
    out_layer: Linear,
}

impl TextEmbedder {
    fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size();

        // ZeroCenterRMSNorm: checkpoint stores (scale - 1), effective scale = weight + 1.
        // We add 1 to the loaded weight so RmsNorm uses the true scale.
        let raw_weight = vb.get(hidden, "text_norm.weight")?;
        let ones = Tensor::ones(raw_weight.shape(), DType::F32, vb.device())?;
        let text_norm_weight = raw_weight.broadcast_add(&ones)?;
        let text_norm = RmsNorm::new(text_norm_weight, 1e-6);

        let in_layer = candle_nn::linear_b(cfg.joint_attention_dim, hidden, false, vb.pp("in_layer"))?;
        let out_layer = candle_nn::linear_b(hidden, hidden, false, vb.pp("out_layer"))?;

        Ok(Self { text_norm, in_layer, out_layer })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // text_norm -> in_layer -> GELU(tanh) -> out_layer
        let x = x.apply(&self.text_norm)?;
        let x = x.apply(&self.in_layer)?;
        let x = x.gelu()?;
        x.apply(&self.out_layer)
    }
}

impl QwenImageTransformer {
    pub fn new(cfg: &Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        let img_in = candle_nn::linear_b(cfg.in_channels, cfg.hidden_size(), false, vb.pp("img_in"))?;

        let txt_in = if vb.contains_tensor("txt_in.in_layer.weight") {
            Some(TextEmbedder::new(cfg, vb.pp("txt_in"))?)
        } else {
            None
        };

        let time_text_embed = QwenTimestepProj::new(cfg, vb.pp("time_text_embed.timestep_embedder"))?;

        let mut transformer_blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = TransformerBlock::new(cfg, vb.pp(format!("transformer_blocks.{}", i)))?;
            transformer_blocks.push(block);
        }

        let final_layer = FinalLayer::new(cfg, vb.pp("norm_out"))?;
        let modulation = Modulation::new(cfg.hidden_size(), vb.pp("modulation"))?;
        let proj_out = candle_nn::linear_b(cfg.hidden_size(), cfg.in_channels, false, vb.pp("proj_out"))?;

        Ok(Self {
            img_in, txt_in, time_text_embed, modulation, transformer_blocks,
            final_layer, proj_out, cfg: cfg.clone(), device,
        })
    }

    /// Full forward pass for denoising step.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        encoder_hidden_states_mask: Option<&Tensor>,
        timestep: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        // Project image
        let x = hidden_states.apply(&self.img_in)?;
        
        // Embed text if provided
        let txt_embedded = if let Some(txt) = encoder_hidden_states {
            if let Some(txt_in) = &self.txt_in {
                Some(txt_in.forward(txt)?)
            } else {
                Some(txt.clone())
            }
        } else {
            None
        };
        // Concatenate image and text along sequence dimension
        let (_b, seq_img, _) = x.dims3()?;
        let has_text = txt_embedded.is_some();
        let seq_txt = if let Some(ref txt) = txt_embedded { txt.dim(1)? } else { 0 };
        let x = if let Some(txt) = txt_embedded {
            Tensor::cat(&[txt, x], 1)?
        } else {
            x
        };
        let t_emb = self.time_text_embed.forward(timestep)?;

        // `causal_condition`: text tokens take a dedicated t=0 modulation row
        // instead of the real, per-step timestep — the model was trained
        // treating the text prefix as noise-free regardless of the image's
        // current diffusion step, while only the target-image tokens are
        // modulated by the actual noise level. The t=0 sinusoidal embedding is
        // the constant [cos(0)=1,...,1, sin(0)=0,...,0] vector (matching
        // scheduler::timestep_embedding's cos-then-sin order), so it can be
        // built directly without a scalar timestep input.
        let hidden = self.cfg.hidden_size();
        let modulation_real = self.modulation.forward(&t_emb)?; // [B, 4*hidden]
        let modulation_real_b = modulation_real.unsqueeze(1)?.broadcast_as((_b, seq_img, 4 * hidden))?;
        let modulation = if seq_txt > 0 {
            const SINUSOIDAL_HALF_DIM: usize = 128; // scheduler::timestep_embedding dim=256
            let zero_sinusoidal = Tensor::cat(
                &[
                    Tensor::ones((1, SINUSOIDAL_HALF_DIM), x.dtype(), &self.device)?,
                    Tensor::zeros((1, SINUSOIDAL_HALF_DIM), x.dtype(), &self.device)?,
                ],
                D::Minus1,
            )?;
            let zero_t_emb = self.time_text_embed.forward(&zero_sinusoidal)?; // [1, hidden]
            let modulation_zero = self.modulation.forward(&zero_t_emb)?; // [1, 4*hidden]
            let modulation_zero_b = modulation_zero
                .unsqueeze(1)?
                .broadcast_as((_b, seq_txt, 4 * hidden))?
                .contiguous()?;
            Tensor::cat(&[&modulation_zero_b, &modulation_real_b.contiguous()?], 1)?
        } else {
            modulation_real_b.contiguous()?
        };

        // Compute RoPE using new QwenImage21Rope API
        // Build image_pad_mask: true for image tokens, false for text (as u8)
        let image_pad_mask: Vec<u8> = vec![0u8; seq_txt].into_iter()
            .chain(vec![1u8; seq_img])
            .collect();
        let image_pad_mask = Tensor::new(image_pad_mask.as_slice(), &self.device)?
            .to_dtype(candle_core::DType::U8)?;
        // VAE scale factor is 16 (AutoencoderKLQwenImage21), patch size is 2
        // patch_size=1: each latent pixel is its own token, no 2x2 packing
        let vae_scale_factor = 16;
        let h_patches = height / vae_scale_factor;
        let w_patches = width / vae_scale_factor;
        let img_shapes = [(1, h_patches, w_patches)];
        let pe = EmbedNd::new(self.cfg.hidden_size(), self.cfg.rope_theta, self.cfg.axes_dims_rope, &self.device)?
            .forward(&img_shapes, &image_pad_mask, &self.device)?;
        let joint_seq_len = x.dim(1)?;
        let (image_ids, target_token_mask, _block_boundaries) = build_token_metadata(&img_shapes, seq_txt, joint_seq_len, &self.device)?;
        let key_valid = if let Some(mask) = encoder_hidden_states_mask {
            // mask: [batch, text_seq_len] -> expand to joint sequence
            let text_mask = mask.to_dtype(candle_core::DType::U8)?;
            // Create ones for image positions
            let batch = text_mask.dim(0)?;
            let img_ones = Tensor::ones((batch, seq_img), candle_core::DType::U8, &self.device)?;
            // Concatenate text mask and image ones
            Some(Tensor::cat(&[text_mask, img_ones], 1)?)
        } else {
            None
        };
        
        // Build block-causal attention mask
        let attention_mask = build_block_causal_mask(&image_ids, key_valid.as_ref(), 1, &self.device)?;
        let mut hidden_states = x;
        for block in &self.transformer_blocks {
            hidden_states = block.forward(
                &hidden_states,
                &modulation,
                &pe,
                Some(&attention_mask),
                Some(&target_token_mask),
            )?;
        }
        
        // Final layer: AdaLayerNormContinuous
        let output = self.final_layer.forward(&hidden_states, &t_emb)?;
        // Final projection
        let output = output.apply(&self.proj_out)?;
        
        // Extract image part only
        let output = if seq_txt > 0 {
            output.narrow(1, seq_txt, seq_img)?
        } else {
            output
        };
        
        Ok(output)
    }
}

impl Module for QwenImageTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // This is a placeholder - the actual forward is called via the specific method above
        Ok(xs.clone())
    }
}

/// Simple attention helper (no mask, no RoPE in this helper).
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, _pe: &Tensor) -> Result<Tensor> {
    let scale = (q.dim(3)? as f64).powf(-0.5);
    let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
    let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
    attn_weights.matmul(&v)
}