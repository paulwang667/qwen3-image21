//! Layout of the transformer's joint sequence (upstream
//! `QwenImage21Transformer2DModel.forward`): condition-image tokens replace the
//! prompt's `<|image_pad|>` slots in place, one slot standing for a 2x2 group of
//! latent cells (so 4 latent tokens per slot, in raster order), and the target
//! image's tokens are appended at the end. Text-to-image is the special case
//! with no condition images: `[text ; target]`.
use candle_core::{bail, Result, Tensor};

/// Latent tokens per `<|image_pad|>` slot (one 32x32-px vision patch = 2x2 latent cells).
pub const TOKENS_PER_SLOT: usize = 4;

/// Condition-image inputs to the transformer for one prompt.
pub struct ConditionTokens<'a> {
    /// Packed condition latents `[B, sum(h*w), 64]`, images concatenated in prompt order.
    pub latents: &'a Tensor,
    /// Latent `(h, w)` of each condition image.
    pub shapes: &'a [(usize, usize)],
    /// True at the prompt's `<|image_pad|>` positions (one per prompt token).
    pub text_image_mask: &'a [bool],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    /// Prompt tokens `start..start + len`.
    Text { start: usize, len: usize },
    /// All tokens of condition image `index`.
    Condition { index: usize },
    Target,
}

#[derive(Debug, Clone)]
pub struct JointLayout {
    pub pieces: Vec<Piece>,
    /// Per joint token: true for image tokens (condition and target).
    pub image_pad_mask: Vec<bool>,
    /// Per joint token: -1 for text, else the image block index (condition
    /// images in order, then the target) — upstream `build_token_metadata`.
    pub image_ids: Vec<i64>,
    /// `(frame, height, width)` in latent tokens per image block, target last.
    pub img_shapes: Vec<(usize, usize, usize)>,
    cond_offsets: Vec<usize>,
    /// Tokens before the target block (text and condition images). Their
    /// activations are step-independent under `causal_condition`.
    pub prefix_len: usize,
    pub target_len: usize,
}

impl JointLayout {
    /// `text_image_mask`: true at the prompt's `<|image_pad|>` positions (length
    /// `text_len`); `cond_shapes`: latent `(h, w)` of each condition image, in
    /// prompt order; `target`: latent `(h, w)` of the image being generated.
    pub fn new(text_len: usize, text_image_mask: Option<&[bool]>, cond_shapes: &[(usize, usize)], target: (usize, usize)) -> Result<Self> {
        let no_images = vec![false; text_len];
        let slots = text_image_mask.unwrap_or(&no_images);
        if slots.len() != text_len {
            bail!("text_image_mask has {} entries for {text_len} prompt tokens", slots.len());
        }
        let mut pieces = Vec::new();
        let mut cond_offsets = Vec::with_capacity(cond_shapes.len());
        let (mut i, mut next_image, mut cond_tokens) = (0usize, 0usize, 0usize);
        while i < text_len {
            if !slots[i] {
                let len = slots[i..].iter().take_while(|&&s| !s).count();
                pieces.push(Piece::Text { start: i, len });
                i += len;
                continue;
            }
            // Blocks are delimited by token counts, not by runs of slots, so two
            // adjacent condition images stay separate blocks.
            let Some(&(h, w)) = cond_shapes.get(next_image) else {
                bail!("prompt has more image slots than condition images");
            };
            let needed = h * w / TOKENS_PER_SLOT;
            if h * w % TOKENS_PER_SLOT != 0 || slots[i..].iter().take(needed).filter(|&&s| s).count() != needed {
                bail!("condition image {next_image} ({h}x{w} latents) needs {needed} consecutive image slots");
            }
            pieces.push(Piece::Condition { index: next_image });
            cond_offsets.push(cond_tokens);
            cond_tokens += h * w;
            next_image += 1;
            i += needed;
        }
        if next_image != cond_shapes.len() {
            bail!("{} condition images but only {next_image} found slots in the prompt", cond_shapes.len());
        }
        pieces.push(Piece::Target);

        let mut img_shapes: Vec<(usize, usize, usize)> = cond_shapes.iter().map(|&(h, w)| (1, h, w)).collect();
        img_shapes.push((1, target.0, target.1));
        let (mut image_pad_mask, mut image_ids) = (Vec::new(), Vec::new());
        for piece in &pieces {
            let (len, id) = match *piece {
                Piece::Text { len, .. } => (len, -1),
                Piece::Condition { index } => (cond_shapes[index].0 * cond_shapes[index].1, index as i64),
                Piece::Target => (target.0 * target.1, cond_shapes.len() as i64),
            };
            image_pad_mask.extend(std::iter::repeat(id >= 0).take(len));
            image_ids.extend(std::iter::repeat(id).take(len));
        }
        let target_len = target.0 * target.1;
        let prefix_len = image_ids.len() - target_len;
        Ok(Self { pieces, image_pad_mask, image_ids, img_shapes, cond_offsets, prefix_len, target_len })
    }

    pub fn len(&self) -> usize {
        self.image_ids.len()
    }

    /// Builds the joint hidden states from projected text `[B, text_len, D]`,
    /// projected condition tokens `[B, sum(h*w), D]` (all images concatenated in
    /// order), and the projected target tokens `[B, h*w, D]`.
    pub fn assemble(&self, text: &Tensor, cond: Option<&Tensor>, target: &Tensor) -> Result<Tensor> {
        let mut parts = Vec::with_capacity(self.pieces.len());
        for piece in &self.pieces {
            parts.push(match *piece {
                Piece::Text { start, len } => text.narrow(1, start, len)?,
                Piece::Condition { index } => {
                    let Some(cond) = cond else { bail!("layout has condition images but no condition tokens") };
                    let (h, w) = (self.img_shapes[index].1, self.img_shapes[index].2);
                    cond.narrow(1, self.cond_offsets[index], h * w)?
                }
                Piece::Target => target.clone(),
            });
        }
        Tensor::cat(&parts, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_is_text_then_target() {
        let l = JointLayout::new(3, None, &[], (2, 2)).unwrap();
        assert_eq!(l.pieces, vec![Piece::Text { start: 0, len: 3 }, Piece::Target]);
        assert_eq!(l.image_ids, vec![-1, -1, -1, 0, 0, 0, 0]);
        assert_eq!((l.prefix_len, l.target_len), (3, 4));
    }

    #[test]
    fn condition_slots_expand_in_place() {
        // "a <pad><pad> b" with one 2x4-latent condition image (8 tokens = 2 slots).
        let mask = [false, true, true, false];
        let l = JointLayout::new(4, Some(&mask), &[(2, 4)], (2, 2)).unwrap();
        assert_eq!(
            l.pieces,
            vec![Piece::Text { start: 0, len: 1 }, Piece::Condition { index: 0 }, Piece::Text { start: 3, len: 1 }, Piece::Target]
        );
        let mut ids = vec![-1];
        ids.extend([0; 8]);
        ids.push(-1);
        ids.extend([1; 4]);
        assert_eq!(l.image_ids, ids);
        assert_eq!((l.prefix_len, l.target_len), (10, 4));
    }

    #[test]
    fn adjacent_condition_images_stay_separate_blocks() {
        let mask = [true, true];
        let l = JointLayout::new(2, Some(&mask), &[(2, 2), (2, 2)], (2, 2)).unwrap();
        assert_eq!(l.pieces, vec![Piece::Condition { index: 0 }, Piece::Condition { index: 1 }, Piece::Target]);
        assert_eq!(l.image_ids[..8], [0, 0, 0, 0, 1, 1, 1, 1]);
    }
}
