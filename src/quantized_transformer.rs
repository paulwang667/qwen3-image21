use candle_core::{Module, Result, Tensor, D, DType, Device};
use candle_nn::{LayerNorm, RmsNorm};
use candle_transformers::quantized_nn::{self as qnn, Linear};
use candle_transformers::quantized_var_builder::VarBuilder;
use crate::rope::{apply_rope, EmbedNd};
use crate::scheduler::timestep_embedding;
use crate::attention_mask::{build_token_metadata, build_block_causal_mask};
use crate::transformer::KVCache;

/// Qwen-Image-2.1 config - same as non-quantized.
pub use crate::transformer::Config;

/// Quantized timestep embedding (linear_1 -> silu -> linear_2).
#[derive(Debug, Clone)]
struct QwenTimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
}

impl QwenTimestepEmbedder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let linear_1 = qnn::linear_b(256, cfg.hidden_size(), false, vb.pp("linear_1"))?;
        let linear_2 = qnn::linear_b(cfg.hidden_size(), cfg.hidden_size(), false, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let x = timestep.apply(&self.linear_1)?;
        let x = x.silu()?;
        x.apply(&self.linear_2)
    }
}

/// Quantized text embedder (QwenImage21TextProjection).
/// Order: text_norm(ZeroCenterRMSNorm) -> in_layer -> GELU(tanh) -> out_layer
#[derive(Debug, Clone)]
struct QTextEmbedder {
    text_norm: RmsNorm,
    in_layer: Linear,
    out_layer: Linear,
}

impl QTextEmbedder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // ZeroCenterRMSNorm: the checkpoint stores `scale - 1`, so add 1 back.
        let text_norm_weight = (vb.get(cfg.hidden_size(), "text_norm.weight")?.dequantize(vb.device())? + 1.0)?;
        let text_norm = RmsNorm::new(text_norm_weight, 1e-6);
        let in_layer = qnn::linear_b(cfg.joint_attention_dim, cfg.hidden_size(), false, vb.pp("in_layer"))?;
        let out_layer = qnn::linear_b(cfg.hidden_size(), cfg.hidden_size(), false, vb.pp("out_layer"))?;
        Ok(Self { text_norm, in_layer, out_layer })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.apply(&self.text_norm)?;
        let x = x.apply(&self.in_layer)?;
        let x = x.gelu()?;
        x.apply(&self.out_layer)
    }
}

/// Quantized single-stream Attention.
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
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size();
        let d_head = cfg.attention_head_dim;

        let to_q = qnn::linear_b(cfg.hidden_size(), h_sz, false, vb.pp("to_q"))?;
        let to_k = qnn::linear_b(cfg.hidden_size(), h_sz, false, vb.pp("to_k"))?;
        let to_v = qnn::linear_b(cfg.hidden_size(), h_sz, false, vb.pp("to_v"))?;
        let to_out = qnn::linear_b(h_sz, cfg.hidden_size(), false, vb.pp("to_out.0"))?;

        let norm_q_w = vb.get(d_head, "norm_q.weight")?.dequantize(vb.device())?;
        let norm_q = RmsNorm::new(norm_q_w, 1e-6);
        let norm_k_w = vb.get(d_head, "norm_k.weight")?.dequantize(vb.device())?;
        let norm_k = RmsNorm::new(norm_k_w, 1e-6);

        Ok(Self {
            to_q, to_k, to_v, to_out, norm_q, norm_k,
            heads: cfg.num_attention_heads,
            dim_head: d_head,
            inner_dim: h_sz,
            dropout: 0.0,
        })
    }

    fn forward(&self, x: &Tensor, pe: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;

        let q = x.apply(&self.to_q)?;
        let k = x.apply(&self.to_k)?;
        let v = x.apply(&self.to_v)?;

        let q = q.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?;
        let k = k.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?;
        let v = v.reshape((b, seq, self.heads, self.dim_head))?.transpose(1, 2)?.contiguous()?; // Metal matmul needs contiguous operands

        let q = q.apply(&self.norm_q)?;
        let k = k.apply(&self.norm_k)?;

        let q = apply_rope(&q, pe)?;
        eprintln!("  [Attention] after rope q.shape={:?}", q.shape());
        let k = apply_rope(&k, pe)?;
        eprintln!("  [Attention] after rope k.shape={:?}", k.shape());


        let scale = (self.dim_head as f64).powf(-0.5);
        eprintln!("  [Attention] q.shape={:?}, k.shape={:?}, v.shape={:?}, scale={}", q.shape(), k.shape(), v.shape(), scale);
        let mut attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        eprintln!("  [Attention] after matmul attn_weights.shape={:?}", attn_weights.shape());
        // Apply attention mask: convert bool mask (True=attend) to additive (0/-inf)
        if let Some(mask) = attention_mask {
            let neg_inf = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?.broadcast_as(mask.dims())?.contiguous()?;
            let zero = Tensor::new(0.0f32, attn_weights.device())?.broadcast_as(mask.dims())?.contiguous()?;
            let additive_mask = mask.where_cond(&zero, &neg_inf)?;
            attn_weights = attn_weights.broadcast_add(&additive_mask)?;
        }

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_out = attn_weights.matmul(&v)?;

        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq, self.inner_dim))?;
        attn_out.apply(&self.to_out)
    }

    /// Cached forward pass with KV cache support.
    fn forward_cached(
        &self,
        x: &Tensor,
        pe: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut crate::transformer::KVCache>,
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

        // Apply attention mask: convert bool mask (True=attend) to additive (0/-inf)
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
        attn_out.apply(&self.to_out)
    }
}

/// Quantized gated MLP (SwiGLU): `out(silu(gate_layer(x)) * proj(x))`.
#[derive(Debug, Clone)]
enum MlpInput {
    Separate { proj: Linear, gate_layer: Linear },
    /// ComfyUI-style conversions (e.g. unsloth's GGUFs) fuse both into one
    /// `gate_up` matmul whose output is `[gate; up]` (ComfyUI's `_swiglu_eager`
    /// chunks it that way), so the fused weight is kept quantized and only the
    /// activation is split.
    FusedGateUp(Linear),
}

#[derive(Debug, Clone)]
struct GatedMlp {
    input: MlpInput,
    out: Linear,
}

impl GatedMlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size();
        let mlp_hidden = cfg.mlp_hidden_size();

        // Not `contains_key`: candle's quantized VarBuilder ignores the `pp` prefix there.
        let input = if vb.get_no_shape("gate_up.weight").is_ok() {
            MlpInput::FusedGateUp(qnn::linear_b(hidden, 2 * mlp_hidden, false, vb.pp("gate_up"))?)
        } else {
            MlpInput::Separate {
                proj: qnn::linear_b(hidden, mlp_hidden, false, vb.pp("proj"))?,
                gate_layer: qnn::linear_b(hidden, mlp_hidden, false, vb.pp("gate_layer"))?,
            }
        };
        let out = qnn::linear_b(mlp_hidden, hidden, false, vb.pp("out"))?;

        Ok(Self { input, out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (gate, proj) = match &self.input {
            MlpInput::Separate { proj, gate_layer } => (x.apply(gate_layer)?, x.apply(proj)?),
            MlpInput::FusedGateUp(gate_up) => {
                let gu = x.apply(gate_up)?;
                let half = gu.dim(D::Minus1)? / 2;
                (gu.narrow(D::Minus1, 0, half)?, gu.narrow(D::Minus1, half, half)?)
            }
        };
        let gated = gate.silu()?.broadcast_mul(&proj)?;
        gated.apply(&self.out)
    }
}

impl Module for GatedMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

/// Global Modulation layer for Qwen-Image-2.1 (quantized).
/// Produces 4 outputs: [scale1, gate1, scale2, gate2]
#[derive(Debug, Clone)]
struct Modulation {
    lin: Linear,
}

impl Modulation {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = qnn::linear_b(dim, 4 * dim, false, vb.pp("1"))?;
        Ok(Self { lin })
    }

    fn forward(&self, t_emb: &Tensor) -> Result<Tensor> {
        t_emb.silu()?.apply(&self.lin)
    }
}

fn layer_norm_no_bias(dim: usize, device: &Device) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, DType::F32, device)?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

/// Quantized single-stream transformer block.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: GatedMlp,
}

impl TransformerBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size();
        let norm1 = layer_norm_no_bias(hidden, vb.device())?;
        let attn = Attention::new(cfg, vb.pp("attn"))?;
        let norm2 = layer_norm_no_bias(hidden, vb.device())?;
        let mlp = GatedMlp::new(cfg, vb.pp("img_mlp"))?;
        Ok(Self { norm1, attn, norm2, mlp })
    }

    fn forward(&self, x: &Tensor, modulation: &Tensor, pe: &Tensor, attention_mask: Option<&Tensor>, _modulation_row_indices: Option<&Tensor>) -> Result<Tensor> {
        // Per-token modulation [B, seq, 4*hidden] -> [scale1, gate1, scale2, gate2]
        let parts = modulation.chunk(4, D::Minus1)?;
        let scale1 = parts[0].clone();
        let gate1 = parts[1].clone();
        let scale2 = parts[2].clone();
        let gate2 = parts[3].clone();
        eprintln!("  [Block] modulation.shape={:?}", modulation.shape());
        eprintln!("  [Block] scale1.shape={:?}, gate1.shape={:?}, scale2.shape={:?}, gate2.shape={:?}", scale1.shape(), gate1.shape(), scale2.shape(), gate2.shape());

        // Attention: scale1 * norm1(x), gate1.tanh() * attn
        let normed = self.norm1.forward(x)?;
        let scale1_u = scale1.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale1_u)?;
        let attn_out = self.attn.forward(&normed, pe, attention_mask)?;
        let gate1_u = gate1.tanh()?;
        let attn_out = gate1_u.broadcast_mul(&attn_out)?;
        let x = x.broadcast_add(&attn_out)?;

        // MLP: scale2 * norm2(x), gate2.tanh() * mlp
        let normed = self.norm2.forward(&x)?;
        let scale2_u = scale2.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale2_u)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let gate2_u = gate2.tanh()?;
        let mlp_out = gate2_u.broadcast_mul(&mlp_out)?;
        let x = x.broadcast_add(&mlp_out)?;

        Ok(x)
    }

    /// Cached forward pass with KV cache support.
    fn forward_cached(
        &self,
        x: &Tensor,
        modulation: &Tensor,
        pe: &Tensor,
        attention_mask: Option<&Tensor>,
        _modulation_row_indices: Option<&Tensor>,
        kv_caches: &mut [crate::transformer::KVCache],
    ) -> Result<Tensor> {
        // Per-token modulation [B, seq, 4*hidden] -> [scale1, gate1, scale2, gate2]
        let parts = modulation.chunk(4, D::Minus1)?;
        let scale1 = parts[0].clone();
        let gate1 = parts[1].clone();
        let scale2 = parts[2].clone();
        let gate2 = parts[3].clone();

        // Attention: scale1 * norm1(x), gate1.tanh() * attn
        let normed = self.norm1.forward(x)?;
        let scale1_u = scale1.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale1_u)?;

        let attn_out = if let Some(cache) = kv_caches.get_mut(0) {
            self.attn.forward_cached(&normed, pe, attention_mask, Some(cache))?
        } else {
            self.attn.forward_cached(&normed, pe, attention_mask, None)?
        };

        let gate1_u = gate1.tanh()?;
        let attn_out = gate1_u.broadcast_mul(&attn_out)?;
        let x = x.broadcast_add(&attn_out)?;

        // MLP: scale2 * norm2(x), gate2.tanh() * mlp
        let normed = self.norm2.forward(&x)?;
        let scale2_u = scale2.affine(1.0, 1.0)?;
        let normed = normed.broadcast_mul(&scale2_u)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let gate2_u = gate2.tanh()?;
        let mlp_out = gate2_u.broadcast_mul(&mlp_out)?;
        let x = x.broadcast_add(&mlp_out)?;

        Ok(x)
    }
}

/// Quantized final layer (AdaLayerNormContinuous, scale-only).
/// norm_out: norm(x) * (1 + linear(silu(temb)))
/// proj_out is separate at the top level of the transformer.
#[derive(Debug, Clone)]
struct FinalLayer {
    norm: LayerNorm,
    linear: Linear,
}

impl FinalLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size();
        let norm = layer_norm_no_bias(hidden, vb.device())?;
        let linear = qnn::linear_b(hidden, hidden, false, vb.pp("linear"))?;
        Ok(Self { norm, linear })
    }

    fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(x)?;
        let scale = self.linear.forward(&t_emb.silu()?)?;
        normed.broadcast_mul(&scale.unsqueeze(1)?.affine(1.0, 1.0)?)
    }
}

/// Quantized Qwen-Image-2.1 Transformer (32 Single-Stream DiT).
#[derive(Debug, Clone)]
pub struct QwenImageTransformerQuantized {
    img_in: Linear,
    txt_in: Option<QTextEmbedder>,
    time_text_embed: QwenTimestepEmbedder,
    modulation: Modulation,
    pub transformer_blocks: Vec<TransformerBlock>,
    final_layer: FinalLayer,
    proj_out: Linear,
    cfg: Config,
    device: Device,
}

impl QwenImageTransformerQuantized {
    /// `causal_condition` modulation `[B, seq_txt + seq_img, 4*hidden]`: text
    /// tokens read a row computed at t=0, image tokens the real timestep's row.
    /// See `QwenImageTransformer::forward` for the non-quantized equivalent.
    fn per_token_modulation(&self, t_emb: &Tensor, seq_txt: usize, seq_img: usize, dtype: DType) -> Result<Tensor> {
        let batch = t_emb.dim(0)?;
        let width = 4 * self.cfg.hidden_size();
        let real = self.modulation.forward(t_emb)?.unsqueeze(1)?.broadcast_as((batch, seq_img, width))?.contiguous()?;
        if seq_txt == 0 {
            return Ok(real);
        }
        // t=0 sinusoidal embedding is the constant [cos(0)=1.., sin(0)=0..] (dim 256).
        let zero_sinusoidal = Tensor::cat(
            &[Tensor::ones((1, 128), dtype, &self.device)?, Tensor::zeros((1, 128), dtype, &self.device)?],
            D::Minus1,
        )?;
        let zero = self
            .modulation
            .forward(&self.time_text_embed.forward(&zero_sinusoidal)?)?
            .unsqueeze(1)?
            .broadcast_as((batch, seq_txt, width))?
            .contiguous()?;
        Tensor::cat(&[&zero, &real], 1)
    }

    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        let img_in = qnn::linear_b(cfg.in_channels, cfg.hidden_size(), false, vb.pp("img_in"))?;

        // Not `contains_key`: candle's quantized VarBuilder ignores the `pp` prefix there.
        let txt_in = if vb.get_no_shape("txt_in.in_layer.weight").is_ok() {
            Some(QTextEmbedder::new(cfg, vb.pp("txt_in"))?)
        } else {
            None
        };

        let time_text_embed = QwenTimestepEmbedder::new(cfg, vb.pp("time_text_embed.timestep_embedder"))?;
        let mut transformer_blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = TransformerBlock::new(cfg, vb.pp(format!("transformer_blocks.{}", i)))?;
            transformer_blocks.push(block);
        }

        let final_layer = FinalLayer::new(cfg, vb.pp("norm_out"))?;
        let modulation = Modulation::new(cfg.hidden_size(), vb.pp("modulation"))?;
        let proj_out = qnn::linear_b(cfg.hidden_size(), cfg.in_channels, false, vb.pp("proj_out"))?;

        Ok(Self {
            img_in, txt_in, time_text_embed, modulation, transformer_blocks,
            final_layer, proj_out, cfg: cfg.clone(), device,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        text_emb: Option<&Tensor>,
        encoder_hidden_states_mask: Option<&Tensor>,
        timestep: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        eprintln!("  [quantized forward] hidden_states.shape={:?}, timestep.shape={:?}", hidden_states.shape(), timestep.shape());
        // Project image
        // Project image
        let x = hidden_states.apply(&self.img_in)?;

        // Embed text
        let txt_embedded = if let Some(txt) = text_emb {
            if let Some(txt_in) = &self.txt_in {
                Some(txt_in.forward(txt)?)
            } else {
                Some(txt.clone())
            }
        } else {
            None
        };

        // Concatenate [txt, img]
        let has_text = txt_embedded.is_some();
        let seq_txt = if let Some(ref txt) = txt_embedded { txt.dim(1)? } else { 0 };
        let seq_img = x.dim(1)?;
        let x = if let Some(txt) = txt_embedded {
            Tensor::cat(&[txt, x], 1)?
        } else {
            x
        };

        // Timestep
        let t_emb = self.time_text_embed.forward(timestep)?;

        let modulation = self.per_token_modulation(&t_emb, seq_txt, seq_img, x.dtype())?;
        eprintln!("  [quantized forward] modulation.shape={:?}, x.shape={:?}", modulation.shape(), x.shape());

        // RoPE using new QwenImage21Rope API
        let image_pad_mask: Vec<u8> = vec![0u8; seq_txt].into_iter()
            .chain(vec![1u8; seq_img])
            .collect();
        let image_pad_mask = Tensor::new(image_pad_mask.as_slice(), &self.device)?
            .to_dtype(candle_core::DType::U8)?;
        // img_shapes: (frame, height_patches, width_patches) for image blocks
        // VAE scale factor is 16 (AutoencoderKLQwenImage21), patch size is 2
        // patch_size=1: each latent pixel is its own token, no 2x2 packing
        let vae_scale_factor = 16;
        let h_patches = height / vae_scale_factor;
        let w_patches = width / vae_scale_factor;
        let img_shapes = [(1, h_patches, w_patches)];
        let pe = EmbedNd::new(self.cfg.hidden_size(), self.cfg.rope_theta, self.cfg.axes_dims_rope, &self.device)?
            .forward(&img_shapes, &image_pad_mask, &self.device)?;

        // Build token metadata for attention mask and modulation row selection
        let joint_seq_len = x.dim(1)?;
        let (image_ids, target_token_mask, _block_boundaries) = build_token_metadata(&img_shapes, seq_txt, joint_seq_len, &self.device)?;

        // Build key_valid mask from encoder_hidden_states_mask (if available)
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

        // Blocks
        let mut h = x;
        for block in &self.transformer_blocks {
            h = block.forward(&h, &modulation, &pe, Some(&attention_mask), Some(&target_token_mask))?;
        }

        // Final layer
        let output = self.final_layer.forward(&h, &t_emb)?;
        // Final projection
        let output = output.apply(&self.proj_out)?;

        // Extract image part
        let output = if seq_txt > 0 {
            output.narrow(1, seq_txt, seq_img)?
        } else {
            output
        };

        Ok(output)
    }

    /// Cached forward pass with KV cache support.
    pub fn forward_cached(
        &self,
        hidden_states: &Tensor,
        text_emb: Option<&Tensor>,
        encoder_hidden_states_mask: Option<&Tensor>,
        timestep: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        height: usize,
        width: usize,
        kv_caches: &mut [crate::transformer::KVCache],
    ) -> Result<Tensor> {
        // Project image
        let x = hidden_states.apply(&self.img_in)?;

        // Embed text
        let txt_embedded = if let Some(txt) = text_emb {
            if let Some(txt_in) = &self.txt_in {
                Some(txt_in.forward(txt)?)
            } else {
                Some(txt.clone())
            }
        } else {
            None
        };

        // Concatenate [txt, img]
        let has_text = txt_embedded.is_some();
        let seq_txt = if let Some(ref txt) = txt_embedded { txt.dim(1)? } else { 0 };
        let seq_img = x.dim(1)?;
        let x = if let Some(txt) = txt_embedded {
            Tensor::cat(&[txt, x], 1)?
        } else {
            x
        };

        // Timestep
        let t_emb = self.time_text_embed.forward(timestep)?;

        let modulation = self.per_token_modulation(&t_emb, seq_txt, seq_img, x.dtype())?;

        // RoPE using new QwenImage21Rope API
        let image_pad_mask: Vec<u8> = vec![0u8; seq_txt].into_iter()
            .chain(vec![1u8; seq_img])
            .collect();
        let image_pad_mask = Tensor::new(image_pad_mask.as_slice(), &self.device)?
            .to_dtype(candle_core::DType::U8)?;
        // img_shapes: (frame, height_patches, width_patches) for image blocks
        // VAE scale factor is 16 (AutoencoderKLQwenImage21), patch size is 2
        // patch_size=1: each latent pixel is its own token, no 2x2 packing
        let vae_scale_factor = 16;
        let h_patches = height / vae_scale_factor;
        let w_patches = width / vae_scale_factor;
        let img_shapes = [(1, h_patches, w_patches)];
        let pe = EmbedNd::new(self.cfg.hidden_size(), self.cfg.rope_theta, self.cfg.axes_dims_rope, &self.device)?
            .forward(&img_shapes, &image_pad_mask, &self.device)?;

        // Build token metadata for attention mask and modulation row selection
        let joint_seq_len = x.dim(1)?;
        let (image_ids, target_token_mask, _block_boundaries) = build_token_metadata(&img_shapes, seq_txt, joint_seq_len, &self.device)?;

        // Build key_valid mask from encoder_hidden_states_mask (if available)
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

        // Blocks
        let mut h = x;
        for block in &self.transformer_blocks {
            h = block.forward(&h, &modulation, &pe, Some(&attention_mask), Some(&target_token_mask))?;
        }

        // Final layer
        let output = self.final_layer.forward(&h, &t_emb)?;
        // Final projection
        let output = output.apply(&self.proj_out)?;

        // Extract image part
        let output = if seq_txt > 0 {
            output.narrow(1, seq_txt, seq_img)?
        } else {
            output
        };

        Ok(output)
    }
}

/// Simple attention helper.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, _pe: &Tensor) -> Result<Tensor> {
    let b = q.dim(0)?;
    let h = q.dim(1)?;
    let seq_q = q.dim(2)?;
    let seq_k = k.dim(2)?;
    let d = q.dim(3)?;

    let scale = (d as f64).powf(-0.5);
    let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
    let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
    let attn_out = attn_weights.matmul(&v)?;

    let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_q, h * d))?;
    Ok(attn_out)
}