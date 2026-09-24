use candle_core::{Module, Result, Tensor, D, DType, Device};
use candle_nn::{LayerNorm, Linear, RmsNorm};
use super::attention_mask::build_block_causal_mask;
use super::joint_layout::{ConditionTokens, JointLayout};
use super::rope::{apply_rope, EmbedNd};
use super::scheduler::TimeEmbedding;

/// Per-layer attention keys/values of the text prefix, reused across
/// denoising steps. Valid because the checkpoint uses `causal_condition`: text
/// tokens modulate from t=0 and attend only (causally) to other text tokens, so
/// their activations never depend on the timestep or the image latents. The
/// first step runs the full sequence and fills this; later steps process only
/// the image tokens, attending to `[cached text K/V ; image K/V]`.
#[derive(Debug, Clone)]
pub struct TextKvCache {
    /// One `(k, v)` per block, each `[batch, heads, text_len, head_dim]`.
    pub(crate) layers: Vec<(Tensor, Tensor)>,
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
        })
    }

    /// `prefix`: cached text-prefix `(k, v)` prepended to this call's keys and
    /// values. Returns the output plus this call's own `(k, v)` (after Q/K norm
    /// and RoPE) so the caller can cache the text part.
    fn forward(
        &self,
        x: &Tensor,
        pe: &Tensor,
        attention_mask: Option<&Tensor>,
        prefix: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
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
        let (k_all, v_all) = match prefix {
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            None => (k.clone(), v.clone()),
        };
        let attn_out = chunked_attention(&q, &k_all, &v_all, (self.dim_head as f64).powf(-0.5), attention_mask)?;

        // Reshape back: [B, H, S, D] -> [B, S, D]
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq, self.inner_dim))?;
        
        // Output projection
        Ok((attn_out.apply(&self.to_out)?, k, v))
    }
    
}

/// Query rows per attention chunk. Image-conditioned sequences reach ~8k
/// tokens at 1024², where one full `[1, 32, S, S]` F32 score matrix is ~8.6 GB;
/// chunking the queries keeps each materialised block near 1 GB. Every query
/// row is computed exactly as without chunking.
const ATTENTION_CHUNK_ROWS: usize = 1024;

/// `softmax(q kᵀ · scale + mask) v` for `q` `[B, H, Sq, D]` and `k`/`v`
/// `[B, H, Skv, D]`. `mask` is a bool/u8 `[B, 1, Sq, Skv]` (or `[B, 1, 1, Skv]`)
/// tensor, true where attending is allowed.
pub(crate) fn chunked_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, mask: Option<&Tensor>) -> Result<Tensor> {
    let sq = q.dim(2)?;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let mut outs = Vec::with_capacity(sq.div_ceil(ATTENTION_CHUNK_ROWS));
    for start in (0..sq).step_by(ATTENTION_CHUNK_ROWS) {
        let len = ATTENTION_CHUNK_ROWS.min(sq - start);
        let q_chunk = q.narrow(2, start, len)?.contiguous()?;
        let mut w = q_chunk.matmul(&kt)?.affine(scale, 0.0)?;
        if let Some(mask) = mask {
            let m = if mask.dim(2)? == 1 { mask.clone() } else { mask.narrow(2, start, len)? };
            // Additive mask: 0 where allowed, -inf where masked.
            let neg_inf = Tensor::new(f32::NEG_INFINITY, w.device())?.to_dtype(w.dtype())?.broadcast_as(m.dims())?;
            let zero = Tensor::new(0f32, w.device())?.to_dtype(w.dtype())?.broadcast_as(m.dims())?;
            w = w.broadcast_add(&m.where_cond(&zero, &neg_inf)?)?;
        }
        let w = candle_nn::ops::softmax(&w, D::Minus1)?;
        outs.push(w.matmul(v)?);
    }
    Tensor::cat(&outs, 2)
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
        prefix: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
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
        let (attn_out, k, v) = self.attn.forward(&normed, pe, attention_mask, prefix)?;
        let gate1_u = gate1.tanh()?;
        let x = x.broadcast_add(&gate1_u.broadcast_mul(&attn_out)?)?;

        // norm2(x) * (1 + scale2), gate2
        let normed = self.norm2.forward(&x)?;
        let scale2_u = scale2.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale2_u)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let gate2_u = gate2.tanh()?;
        let x = x.broadcast_add(&gate2_u.broadcast_mul(&mlp_out)?)?;

        Ok((x, k, v))
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
        _img_ids: &Tensor,
        _txt_ids: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        self.forward_conditioned(hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep, height, width, None, None)
    }

    /// Denoising step reusing the text prefix's per-layer K/V (see
    /// [`TextKvCache`]). With `*cache == None` this runs the full sequence and
    /// fills the cache; afterwards only the image tokens are processed. The
    /// cache is tied to one prompt embedding — use a separate one per prompt.
    pub fn forward_with_text_cache(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_hidden_states_mask: Option<&Tensor>,
        timestep: &Tensor,
        height: usize,
        width: usize,
        cache: &mut Option<TextKvCache>,
    ) -> Result<Tensor> {
        self.forward_conditioned(hidden_states, Some(encoder_hidden_states), encoder_hidden_states_mask, timestep, height, width, None, Some(cache))
    }

    /// General denoising step: `hidden_states` are the target's packed noisy
    /// latents; `condition` places condition images in the prompt's image
    /// slots (see [`JointLayout`]); `cache` reuses the prefix K/V (text and
    /// condition images) as in [`Self::forward_with_text_cache`]. Returns the
    /// target tokens' prediction.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_conditioned(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        encoder_hidden_states_mask: Option<&Tensor>,
        timestep: &Tensor,
        height: usize,
        width: usize,
        condition: Option<&ConditionTokens>,
        cache: Option<&mut Option<TextKvCache>>,
    ) -> Result<Tensor> {
        let target = hidden_states.apply(&self.img_in)?;
        let (batch, seq_img, _) = target.dims3()?;
        let seq_txt = encoder_hidden_states.map_or(Ok(0), |t| t.dim(1))?;
        // patch_size=1: each latent pixel is its own token, no 2x2 packing
        let vae_scale_factor = 16;
        let layout = JointLayout::new(
            seq_txt,
            condition.map(|c| c.text_image_mask),
            condition.map_or(&[][..], |c| c.shapes),
            (height / vae_scale_factor, width / vae_scale_factor),
        )?;
        let prefix_len = layout.prefix_len;
        let cached = cache.as_ref().and_then(|c| c.as_ref()).cloned();
        let extract = prefix_len > 0 && cache.is_some() && cached.is_none();

        // With a cached prefix, text and condition tokens are not recomputed at all.
        let x = match (encoder_hidden_states, &cached) {
            (Some(txt), None) => {
                let txt = match &self.txt_in {
                    Some(txt_in) => txt_in.forward(txt)?,
                    None => txt.clone(),
                };
                let cond = condition.map(|c| c.latents.apply(&self.img_in)).transpose()?;
                layout.assemble(&txt, cond.as_ref(), &target)?
            }
            _ => target,
        };
        let t_emb = self.time_text_embed.forward(timestep)?;

        // `causal_condition`: text and condition-image tokens take a dedicated
        // t=0 modulation row instead of the real, per-step timestep — the model
        // was trained treating them as noise-free regardless of the target's
        // current diffusion step, while only the target-image tokens are
        // modulated by the actual noise level. The t=0 sinusoidal embedding is
        // the constant [cos(0)=1,...,1, sin(0)=0,...,0] vector (matching
        // scheduler::timestep_embedding's cos-then-sin order), so it can be
        // built directly without a scalar timestep input.
        let hidden = self.cfg.hidden_size();
        let modulation_real = self.modulation.forward(&t_emb)?; // [B, 4*hidden]
        let modulation_real_b = modulation_real.unsqueeze(1)?.broadcast_as((batch, seq_img, 4 * hidden))?;
        let modulation = if prefix_len > 0 && cached.is_none() {
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
                .broadcast_as((batch, prefix_len, 4 * hidden))?
                .contiguous()?;
            Tensor::cat(&[&modulation_zero_b, &modulation_real_b.contiguous()?], 1)?
        } else {
            modulation_real_b.contiguous()?
        };

        // RoPE over the full joint layout; a cached step keeps only the target rows.
        let image_pad_mask: Vec<u8> = layout.image_pad_mask.iter().map(|&m| m as u8).collect();
        let image_pad_mask = Tensor::new(image_pad_mask.as_slice(), &self.device)?;
        let pe = EmbedNd::new(self.cfg.hidden_size(), self.cfg.rope_theta, self.cfg.axes_dims_rope, &self.device)?
            .forward(&layout.img_shapes, &image_pad_mask, &self.device)?;
        let pe = if cached.is_some() { pe.narrow(0, prefix_len, seq_img)?.contiguous()? } else { pe };

        // Keys cover the whole joint sequence in both modes; only prompt
        // (text) positions can be padding.
        let key_valid = match encoder_hidden_states_mask {
            Some(mask) => {
                let mask = mask.to_dtype(candle_core::DType::U8)?;
                let b = mask.dim(0)?;
                let cond_ones = Tensor::ones((b, layout.len() - seq_txt - seq_img), candle_core::DType::U8, &self.device)?;
                let target_ones = Tensor::ones((b, seq_img), candle_core::DType::U8, &self.device)?;
                Some(layout.assemble(&mask, Some(&cond_ones), &target_ones)?)
            }
            None => None,
        };
        let attention_mask = if cached.is_some() {
            // Target queries see every (valid) prefix key and every target key.
            match &key_valid {
                Some(kv) => Some(kv.unsqueeze(1)?.unsqueeze(1)?),
                None => None,
            }
        } else {
            let image_ids = Tensor::new(layout.image_ids.as_slice(), &self.device)?;
            Some(build_block_causal_mask(&image_ids, key_valid.as_ref(), batch, &self.device)?)
        };

        let mut hidden_states = x;
        let mut extracted = Vec::with_capacity(if extract { self.transformer_blocks.len() } else { 0 });
        for (i, block) in self.transformer_blocks.iter().enumerate() {
            let prefix = cached.as_ref().map(|c| &c.layers[i]);
            let (h, k, v) = block.forward(&hidden_states, &modulation, &pe, attention_mask.as_ref(), prefix)?;
            if extract {
                extracted.push((k.narrow(2, 0, prefix_len)?.contiguous()?, v.narrow(2, 0, prefix_len)?.contiguous()?));
            }
            hidden_states = h;
        }
        if extract {
            if let Some(cache) = cache {
                *cache = Some(TextKvCache { layers: extracted });
            }
        }

        // Final layer: AdaLayerNormContinuous, then projection; keep target tokens.
        let output = self.final_layer.forward(&hidden_states, &t_emb)?.apply(&self.proj_out)?;
        if cached.is_none() && prefix_len > 0 {
            output.narrow(1, prefix_len, seq_img)
        } else {
            Ok(output)
        }
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