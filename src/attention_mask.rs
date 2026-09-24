//! Block-causal attention mask for Qwen-Image-2.1
//!
//! Implements the mask: (q_idx >= kv_idx) OR (same_image_block AND q_image_id >= 0)
//! with key_valid masking for padded text tokens.

use candle_core::{Result, Tensor, D, Device};

/// Image id for each token in the joint sequence.
/// -1 for text tokens, unique non-negative id per image block.
pub type ImageIds = Tensor; // [seq_len] i64

/// Mask for valid keys: [batch, seq_len] bool, false at padded text positions.
pub type KeyValid = Tensor; // [batch, seq_len] bool

/// Block-causal attention mask for prefill (dense bool mask) with block boundary support.
/// Returns [batch, 1, seq_len_q, seq_len_kv] bool mask suitable for SDPA.
/// True = attend, False = masked.
pub fn build_block_causal_mask(
    image_ids: &Tensor,           // [seq_len] i64
    key_valid: Option<&Tensor>,   // [batch, seq_len] bool or None
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = image_ids.dim(0)?;
    let dtype = image_ids.dtype();

    // image_ids: [seq_len] -> [seq_len_q, seq_len_kv] broadcast
    let q_image_ids = image_ids.unsqueeze(1)?.expand((seq_len, seq_len))?; // [seq_len, seq_len]
    let kv_image_ids = image_ids.unsqueeze(0)?.expand((seq_len, seq_len))?; // [seq_len, seq_len]

    // same_image_block: (q_image_id == kv_image_id) & (q_image_id >= 0)
    let eq = q_image_ids.eq(&kv_image_ids)?; // [seq_len, seq_len] bool
    let ge_zero = q_image_ids.ge(&Tensor::zeros((seq_len, seq_len), dtype, device)?)?; // [seq_len, seq_len] bool
    // For bool tensors: AND = multiply
    let same_block = eq.mul(&ge_zero)?; // bool AND = multiply

    // causal: q_idx >= kv_idx
    let q_idx = Tensor::arange(0, seq_len as i64, device)?
        .unsqueeze(1)?
        .expand((seq_len, seq_len))?; // [seq_len, seq_len]
    let kv_idx = Tensor::arange(0, seq_len as i64, device)?
        .unsqueeze(0)?
        .expand((seq_len, seq_len))?; // [seq_len, seq_len]
    let causal = q_idx.ge(&kv_idx)?; // [seq_len, seq_len] bool

    // allowed = causal | same_block (OR = max for bool: max(a, b))
    let allowed = causal.maximum(&same_block)?; // [seq_len, seq_len] bool

    // Apply key_valid: [batch, seq_len_kv] -> [batch, 1, 1, seq_len_kv]
    let mut mask = allowed
        .unsqueeze(0)?      // [1, seq_len, seq_len]
        .unsqueeze(0)?;     // [1, 1, seq_len, seq_len]

    if let Some(kv) = key_valid {
        // kv: [batch, seq_len] -> [batch, 1, 1, seq_len]
        let kv_mask = kv
            .unsqueeze(1)?    // [batch, 1, seq_len]
            .unsqueeze(1)?;   // [batch, 1, 1, seq_len]
        mask = mask.mul(&kv_mask)?; // [batch, 1, seq_len, seq_len]
    } else {
        // expand to batch
        mask = mask.expand((batch_size, 1, seq_len, seq_len))?;
    }

    Ok(mask)
}
