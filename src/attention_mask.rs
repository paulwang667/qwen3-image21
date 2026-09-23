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

/// Build token metadata: image_ids and target_token_mask from img_shapes.
/// Supports text prefix followed by multiple image blocks (condition + target).
///
/// Args:
///   - `img_shapes`: list of (frame, height, width) per image, condition images first, target last.
///   - `text_seq_len`: length of text prefix tokens.
///   - `total_seq_len`: total sequence length (text + all image tokens).
/// Returns: (image_ids [seq_len], target_token_mask [seq_len], block_boundaries [Vec<usize>])
///   - image_ids: -1 for text, 0...N for image blocks
///   - target_token_mask: 1 for target image tokens, 0 otherwise
///   - block_boundaries: start index of each image block (relative to full sequence)
pub fn build_token_metadata(
    img_shapes: &[(usize, usize, usize)],
    text_seq_len: usize,
    total_seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<usize>)> {
    let mut image_ids_vec: Vec<i64> = Vec::with_capacity(total_seq_len);
    let mut target_token_mask_vec: Vec<u8> = Vec::with_capacity(total_seq_len);
    let mut block_boundaries: Vec<usize> = Vec::with_capacity(img_shapes.len() + 1);
    
    // Text prefix
    for _ in 0..text_seq_len {
        image_ids_vec.push(-1i64);
        target_token_mask_vec.push(0u8);
    }
    block_boundaries.push(text_seq_len);
    
    let mut cursor = text_seq_len;
    for (img_idx, &(frame, height, width)) in img_shapes.iter().enumerate() {
        let block_len = frame * height * width;
        let is_target = img_idx == img_shapes.len() - 1;

        for _ in 0..block_len {
            image_ids_vec.push(img_idx as i64);
            target_token_mask_vec.push(if is_target { 1 } else { 0 });
        }
        cursor += block_len;
        block_boundaries.push(cursor);
    }

    if cursor < total_seq_len {
        let pad = total_seq_len - cursor;
        image_ids_vec.extend(std::iter::repeat(-1i64).take(pad));
        target_token_mask_vec.extend(std::iter::repeat(0u8).take(pad));
    }

    image_ids_vec.truncate(total_seq_len);
    target_token_mask_vec.truncate(total_seq_len);

    let image_ids = Tensor::new(image_ids_vec.as_slice(), device)?;
    let target_token_mask = Tensor::new(target_token_mask_vec.as_slice(), device)?;

    Ok((image_ids, target_token_mask, block_boundaries))
}

/// Select modulation rows per token based on target_token_mask.
/// Matches diffusers `_select_modulation_rows` for the case this codebase actually
/// exercises: single-image generation with no condition-image prefix.
///
/// The real function's `causal_condition` path additionally supports a `[batch+1, dim]`
/// modulation (a trailing `t=0` row shared by text/condition-image tokens, with target-image
/// tokens using their own sample's row) whenever a mask is given. Nothing in this codebase
/// ever constructs that extra row — `pipeline::denoise` always produces a plain `[batch, dim]`
/// modulation — so that split is not implemented here; every token uses the same row.
///
/// Args:
///   - `modulation`: [batch, dim]
///   - `target_token_mask`: unused placeholder for the unimplemented causal_condition split
/// Returns: [batch, 1, 1, dim] broadcastable to hidden_states
pub fn select_modulation_rows(
    modulation: &Tensor,
    _target_token_mask: Option<&Tensor>,
) -> Result<Tensor> {
    modulation.unsqueeze(1)?.unsqueeze(1) // [B, 1, 1, dim]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn test_build_token_metadata() {
        let device = Device::Cpu;
        let img_shapes = [(1, 2, 2), (1, 2, 2)]; // 2 condition images, 4 tokens each = 8
        let (ids, target, _boundaries) = build_token_metadata(&img_shapes, 0, 8, &device).unwrap();
        assert_eq!(ids.dim(0).unwrap(), 8);
        assert_eq!(target.dim(0).unwrap(), 8);
    }

    /// Regression test for the text-to-image pipeline's actual call shape: a text
    /// prefix followed by exactly one (target) image block, with `img_shapes` passed
    /// through correctly (previously called with `&[]`, which left every image token's
    /// `image_id` at -1 and silently degenerated the block-causal mask to pure causal).
    #[test]
    fn test_build_token_metadata_single_target_image() {
        let device = Device::Cpu;
        let text_seq_len = 3;
        let img_shapes = [(1, 2, 2)]; // 4 image tokens
        let total_seq_len = text_seq_len + 4;
        let (ids, target, _boundaries) =
            build_token_metadata(&img_shapes, text_seq_len, total_seq_len, &device).unwrap();
        let ids: Vec<i64> = ids.to_vec1().unwrap();
        let target: Vec<u8> = target.to_vec1().unwrap();
        assert_eq!(ids, vec![-1, -1, -1, 0, 0, 0, 0]);
        assert_eq!(target, vec![0, 0, 0, 1, 1, 1, 1]);
    }
}