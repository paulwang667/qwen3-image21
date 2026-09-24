# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch Rust inference engine for Qwen-Image-2.1 (text-to-image MMDiT diffusion model), built directly on `candle-core`/`candle-nn` (not `candle-transformers`' model zoo — only its `quantized_nn`/`quantized_var_builder` utilities are reused for GGUF loading). GPU support is opt-in per build via Cargo features (`metal` for Apple Silicon, `cuda` for NVIDIA/Linux — see Commands); building with neither runs on CPU. `main.rs` tries CUDA then Metal then falls back to CPU at runtime, so the same source works on either platform — just pick the matching feature flag when building.

The model weights the code is written against use a **single-stream** MMDiT architecture (one sequence of concatenated text+image tokens per block, shared modulation across all blocks) — not the classic Flux-style dual-stream (separate `img_attn`/`txt_attn`) design. `list_tensors.rs` and `test_gguf.rs` at the repo root are leftover scratch probes from reverse-engineering the checkpoint's tensor names; they are **not** part of the cargo build (no `[[bin]]` entries reference them) and can be ignored or deleted.

## Commands

```bash
cargo check                    # fast type-check, use this while iterating
cargo build --release          # release build, CPU only (no GPU feature enabled)
cargo build --release --features metal   # macOS
cargo build --release --features cuda    # Linux/NVIDIA
cargo test                     # run unit tests (gguf_mapping, attention_mask)
cargo test test_identity_global   # run a single test by name

# Run inference (requires a real transformer + VAE model on disk; see models/)
cargo run --release -- --prompt "a cat" --model-path models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors --vae-path models/vae/qwen_image_2.1_vae_bf16.safetensors
cargo run --release --features cuda -- --prompt "a cat" --model-path models/qwen-image-2.1-Q8_0.gguf --vae-path <repo>/vae/diffusion_pytorch_model.safetensors --text-encoder-path <repo>/text_encoder
# Optional true CFG (off by default, as upstream — 2.1 is meant to be sampled without guidance; doubles the cost)
cargo run --release -- --prompt "a cat" --model-path <model> --negative-prompt "blurry" --true-cfg-scale 4
cargo run --release -- --prompt "a cat" --model-path <model> --benchmark --benchmark-iterations 5

# Known-good path: official non-quantized weights + official text encoder (Qwen/Qwen-Image-2.1 repo)
cargo run --release --features cuda -- --prompt "a red apple on a wooden table" --height 1024 --width 1024 --steps 20 \
  --model-path <repo>/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  --vae-path <repo>/vae/diffusion_pytorch_model.safetensors --text-encoder-path <repo>/text_encoder
# Omitting --text-encoder-path falls back to random embeddings.

# Inspect a GGUF checkpoint's tensor names/shapes
cargo run --bin list_tensors
# Diagnostics used while debugging real-checkpoint loading (see git log for context)
cargo run --bin vae_probe               # decode synthetic latents directly, bypassing the transformer
cargo run --bin transformer_probe       # single transformer forward pass, measure output diversity
cargo run --bin text_encoder_probe [dir] # confirm the text encoder is prompt-dependent
```

### Verifying against upstream (image-conditioned path)

`tools/ref_dump_i2i.py` (run in a Python env with torch + diffusers main + transformers) dumps golden tensors from the upstream `QwenImage21Pipeline` for one condition image; `cargo run --release --bin i2i_probe -- <stage> <ref_dir>` compares each stage (`vae`, `pre`, `vision`, `text`, `transformer`, `encode`). The dump disables TF32 — PyTorch's cuDNN default otherwise adds ~1e-3 noise to the VAE/vision stages; with it off every stage matches to ~1e-5. Use a condition image already at the target size so neither side resizes.

There is no lint config beyond rustc warnings (`cargo check` surfaces them); there is no CI in this repo.

## Architecture

### Module map (`src/`)

- `main.rs` — CLI (clap) entry point. Orchestrates the pipeline in explicit **phases**, dropping each model before loading the next to control peak memory: (1) text encoder (with vision tower) loaded → prompt encoded, with any `--image` condition images → dropped, (1b) VAE encoder → condition images to latents (only with `--image`), (2) transformer → denoising loop → dropped, (3) VAE decoder → latents decoded → dropped. After each phase `release_freed_memory` synchronizes the device: candle's CUDA backend frees into a stream-ordered pool that only returns memory to the driver on synchronize, so without it the transformer phase's buffers stayed reserved through the VAE decode. `--precision bf16` loads the full-precision transformer and the GPU text encoder in BF16 (GGUF transformers and the VAE stay F32; latents stay F32 across steps and attention softmax is F32); `--text-encoder-cpu` runs the text encoder + vision tower on the CPU in F32. README's Memory table has the measured peaks. With `--image`, `--height/--width` default to the last condition image's size at `--output-resolution` (as upstream). Benchmark mode instead keeps transformer + VAE resident for repeated timed iterations.
- `pipeline.rs` — `TransformerType` enum (`NonQuantized`/`Quantized`) dispatches to whichever variant was loaded. `denoise()` runs the Euler flow-matching loop over packed latents; each `PromptEmbeds` carries its own image-slot mask, and `ConditionLatents` (shared by the positive and CFG-negative prompt) are placed into the sequence via `forward_conditioned`. `decode_latents()` unpacks/denormalizes and calls the VAE. The initial noise is drawn on the host from `--seed` (default random): candle's device RNGs start from a fixed state every process (CUDA/Metal) or can't be seeded (CPU), and identical noise is harmful — editing an image generated from the same noise locks the trajectory onto it (oversaturated "HDR" output, reproduced identically by upstream).
- **Prefix KV cache** (`transformer::TextKvCache`, on by default, `--no-kv-cache` to disable): with `causal_condition` the text and condition-image tokens' per-layer K/V never depend on the timestep or target latents, so `denoise` keeps one cache per prompt (a second for the CFG negative prompt). The first step runs the full sequence and fills it with everything before the target block (`JointLayout::prefix_len`); later steps feed only the target tokens, which attend to `[cached prefix K/V ; target K/V]` without any mask. Measured at 1024² / 20 steps (text-to-image): Q8_0 67.1→58.0 s, full precision 99.2→89.9 s — almost all of it from dropping the per-layer mask ops, not from skipping ~15 text tokens. The cache is stored as `KV_CACHE_DTYPE` (BF16) and cast back to the compute dtype when attended to — with two 1024² references the F32 prefix K/V was ~8.7 GB. `kv_cache_probe` checks exactness (512², cached vs. uncached second step): cosine 0.999997 full precision, 0.9997 Q4_K_M. Before BF16 storage it was ~1e-5 for full precision and ~0.8% for CUDA quantized at 256²+ (candle's quantized matmul numerics depend on the row count).
- `transformer.rs` — full-precision `QwenImageTransformer` (`VarBuilder`-based, safetensors). Defines the single-stream `TransformerBlock` (LayerNorm → shared modulation split into scale/gate pairs → attention → MLP), `Attention` (Q/K RMSNorm + RoPE), `GatedMlp` (SwiGLU: `out(silu(gate_layer(x)) * proj(x))` — the activation is on `gate_layer`, matching upstream `QwenImage21SwiGLUFeedForward`; swapping the branches loads fine but produces pure noise), and `TextEmbedder` (ZeroCenterRMSNorm text projection — checkpoint stores `scale - 1`, code adds 1 back). `Config::default()` describes the actual checkpoint shape (32 layers, 32 heads, head_dim 128 → hidden 4096), matching README.md's Model Requirements section.
- `quantized_transformer.rs` — mirrors `transformer.rs` block-for-block but built on `candle_transformers::quantized_nn`/`quantized_var_builder` so weights stay quantized (no dequantize-to-F32 step) during the forward pass. Loads both the unfused naming and ComfyUI-style GGUFs (e.g. `unsloth/Qwen-Image-2.1-GGUF`): `main.rs` strips their `model.diffusion_model.` prefix, and their fused `img_mlp.gate_up` (`[gate; up]`) stays one quantized matmul whose output is split. **candle's quantized `VarBuilder::contains_key` ignores the `pp` prefix** — probe optional tensors with `get_no_shape(..).is_ok()` instead (using `contains_key` once silently dropped `txt_in`). `quant_vs_full_probe` checks a single forward against the full-precision model (expect cosine ≈ 0.9999 at Q8_0).
- `gguf_mapping.rs` — pure name-validation: the checkpoint's GGUF tensor names already match this code's expected names 1:1, so `map_gguf_name` is an identity function with strict allowlisting (rejects anything not matching a known global or per-block tensor pattern). This exists to catch checkpoint format drift early, not to do real renaming.
- `gguf_mapped.rs` — `MappedVarBuilder` wraps `quantized_var_builder::VarBuilder` and applies `gguf_mapping`'s name map when resolving tensors, falling back to the raw name if unmapped.
- `gguf_loader.rs` — standalone dequantize-GGUF-to-safetensors utility (used by `test_gguf.rs`), independent of the `MappedVarBuilder` quantized-load path used by `main.rs --quantized`.
- `rope.rs` — 3-axis (frame/height/width) RoPE (`EmbedNd`), matching diffusers' `QwenImage21Rope`: text tokens advance position on all three axes; each image block freezes the frame axis and lays out a height/width grid centered on zero. Supports `Precomputed` (default) or `Complex` frequency modes.
- `attention_mask.rs` — block-causal attention mask: a query may attend to any key at or before it (causal) OR any key in the same image block (`same_image_block AND image_id >= 0`), so text is causal and each image block is internally bidirectional. The per-token `image_ids` come from `JointLayout`. The checkpoint has `causal_condition: true`: text and condition-image tokens get a modulation row computed at t=0 and only target tokens the real timestep's row, so `TransformerBlock` receives a per-token `[B, seq, 4*hidden]` modulation (both transformers implement this). Attention itself goes through `transformer::chunked_attention` (1,024 query rows at a time), because image-conditioned sequences reach ~8k tokens at 1024², where one full F32 score matrix is ~8.6 GB.
- `scheduler.rs` — `FlowMatchEuler` (the checkpoint's dynamic exponential sigma shift + `shift_terminal` stretch, starting from `linspace(1, 1/N, N)` as the Qwen pipelines do; Euler step) and sinusoidal `timestep_embedding` (`[cos, sin]` order, i.e. `flip_sin_to_cos=True`).
- `vae.rs` — `VaeDecoder` and `VaeEncoder` implementing the real `AutoencoderKLQwenImage21` (64-channel latents, 16× spatial compression, RGBA in/out, real `latents_mean`/`latents_std`). It's a causal-3D VAE (Wan-style) in diffusers, but for single-frame images `QwenImage21CausalConv3d` collapses to a plain `Conv2d`, so both halves are pure 2D; the temporal `time_conv` layers only run from a second frame on and are unused. Two single-frame subtleties are replicated exactly: the decoder's `dup_up3d()` depth-to-space shortcut (its `factor_t` still changes the channel mapping on channel-reducing blocks) and the encoder's `avg_down3d()` shortcut, whose `factor_t=2` blocks average in a front-padded zero frame. The encoder is official-diffusers-layout only and returns the normalized posterior mean (upstream `sample_mode="argmax"`). Stride-1 3×3 convs go through `conv2d_banded`, which convolves row bands (with k−1 halo rows) under a 1 GiB im2col budget — candle's conv2d materialises a `C_in·k·k·H·W` im2col buffer, ~11 GB for the largest decoder layer at 1024². Plus `pack_latents`/`unpack_latents`/`normalize_latents`.
- `text_encoder.rs` — `TextEncoder::load` auto-detects the directory layout. **Official** (a `config.json` with `text_config`, e.g. the official repo's `text_encoder/` with its tokenizer in the sibling `processor/`): the Qwen3-VL language model (`qwen3_vl_text.rs`) plus its vision tower (`qwen3_vl_vision.rs`). `encode` wraps the prompt in the upstream raw template (`<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`), takes the last decoder layer's output **before** the final RMSNorm, and drops the system-prefix tokens — all three are required to match what `txt_in` was trained on. `encode_with_images` uses the image-conditioned template (`<image1><|vision_start|>` + one `<|image_pad|>` per merged 32×32-px vision patch + `<|vision_end|>` before the prompt) and also returns which returned tokens are image slots. **Stand-in** (a plain dense Qwen3 checkpoint, e.g. Qwen3-0.6B): text-only, hidden states tiled up to 4096 — prompt-dependent but semantically wrong.
- `qwen3_vl_text.rs` — the Qwen3-VL language model (vendored: candle-transformers' `qwen3_vl` only exposes last-token logits and has no multimodal RoPE). Implements upstream `get_rope_index` (image tokens get `(start+t, start+h, start+w)` on the merged grid; text resumes at `start + max(h, w)`) with the checkpoint's **interleaved** mRoPE (`mrope_section [24, 20, 20]`), replaces `<|image_pad|>` embeddings with vision features, and adds the DeepStack features to image positions after layers 0–2. Text-only input reduces exactly to 1D RoPE.
- `qwen3_vl_vision.rs` — the Qwen3-VL vision tower, vendored from candle-transformers (private there); the one change is exact `gelu_erf` in the patch mergers, as upstream's `nn.GELU()`.
- `condition_image.rs` — condition-image preprocessing: resize to ~`resolution²` at the image's aspect (multiples of 32, upstream `calculate_dimensions`), RGBA in [-1, 1] for the VAE, and RGB-over-white patches in `Qwen2VLImageProcessor` order for the vision tower. Upstream resizes with PIL Lanczos, which `image`'s Lanczos3 approximates but does not reproduce bit-for-bit.
- `joint_layout.rs` — `JointLayout`: the transformer's joint sequence. Condition-image tokens replace the prompt's `<|image_pad|>` slots in place (4 latent tokens per slot, raster order; blocks delimited by token counts so adjacent images stay separate), the target block is appended; also yields RoPE's image mask, per-token block ids, and `prefix_len`.

### Known incomplete pieces (don't assume otherwise)


- Two independent GGUF-loading code paths exist: `gguf_loader.rs` (dequantize to F32 safetensors, used only by the standalone `test_gguf.rs` probe) vs. `gguf_mapped.rs`/`quantized_transformer.rs` (stay quantized, used by `main.rs --quantized`). Don't conflate them when touching GGUF loading.
- **`main.rs`'s `--quantized` CLI flag is dead**: `load_transformer` routes purely on whether `--model-path` ends in `.gguf`, never reads the `quantized` argument (rustc even flags it as unused).
- **`gguf_mapped::MappedVarBuilder`'s name mapping is bypassed in practice**: `main.rs` calls `mapped_vb.inner().clone()` before constructing `QwenImageTransformerQuantized`, discarding the wrapper. Harmless today because `gguf_mapping::map_gguf_name` is identity-only, but the mapping layer isn't actually exercised by any real code path.

### Models

`models/` holds real checkpoint files (GGUF and safetensors, multi-GB) plus a HuggingFace `.cache` download-lock directory — excluded via `.gitignore` (`/models`, alongside `/target`); don't force-add anything under it.
