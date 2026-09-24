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

There is no lint config beyond rustc warnings (`cargo check` surfaces them); there is no CI in this repo.

## Architecture

### Module map (`src/`)

- `main.rs` — CLI (clap) entry point. Orchestrates the pipeline in three explicit **phases**, dropping each model before loading the next to control peak memory: (1) text encoder loaded → prompt encoded → dropped, (2) transformer loaded → denoising loop run → dropped, (3) VAE loaded → latents decoded → dropped. Benchmark mode instead keeps transformer + VAE resident for repeated timed iterations.
- `pipeline.rs` — `TransformerType` enum (`NonQuantized`/`Quantized`) dispatches `forward()` to whichever variant was loaded. `denoise()` runs the Euler flow-matching loop over packed latents; `decode_latents()` unpacks/denormalizes and calls the VAE.
- **Text-prefix KV cache** (`transformer::TextKvCache`, on by default, `--no-kv-cache` to disable): with `causal_condition` the text tokens' per-layer K/V never depend on the timestep or latents, so `denoise` keeps one cache per prompt (a second for the CFG negative prompt). The first step runs the full sequence and fills it; later steps feed only image tokens, which attend to `[cached text K/V ; image K/V]` without any mask. Measured at 1024² / 20 steps: Q8_0 67.1→58.0 s, full precision 99.2→89.9 s — almost all of it from dropping the per-layer 4116×4116 mask ops, not from skipping ~15 text tokens. `kv_cache_probe` checks exactness: full precision matches uncached to ~1e-5; CUDA quantized is bit-exact at small sizes but ~0.8% off at 256²+ because candle's quantized matmul numerics depend on the row count (still below Q8_0's own error).
- `transformer.rs` — full-precision `QwenImageTransformer` (`VarBuilder`-based, safetensors). Defines the single-stream `TransformerBlock` (LayerNorm → shared modulation split into scale/gate pairs → attention → MLP), `Attention` (Q/K RMSNorm + RoPE), `GatedMlp` (SwiGLU: `out(silu(gate_layer(x)) * proj(x))` — the activation is on `gate_layer`, matching upstream `QwenImage21SwiGLUFeedForward`; swapping the branches loads fine but produces pure noise), and `TextEmbedder` (ZeroCenterRMSNorm text projection — checkpoint stores `scale - 1`, code adds 1 back). `Config::default()` describes the actual checkpoint shape (32 layers, 32 heads, head_dim 128 → hidden 4096), matching README.md's Model Requirements section.
- `quantized_transformer.rs` — mirrors `transformer.rs` block-for-block but built on `candle_transformers::quantized_nn`/`quantized_var_builder` so weights stay quantized (no dequantize-to-F32 step) during the forward pass. Loads both the unfused naming and ComfyUI-style GGUFs (e.g. `unsloth/Qwen-Image-2.1-GGUF`): `main.rs` strips their `model.diffusion_model.` prefix, and their fused `img_mlp.gate_up` (`[gate; up]`) stays one quantized matmul whose output is split. **candle's quantized `VarBuilder::contains_key` ignores the `pp` prefix** — probe optional tensors with `get_no_shape(..).is_ok()` instead (using `contains_key` once silently dropped `txt_in`). `quant_vs_full_probe` checks a single forward against the full-precision model (expect cosine ≈ 0.9999 at Q8_0).
- `gguf_mapping.rs` — pure name-validation: the checkpoint's GGUF tensor names already match this code's expected names 1:1, so `map_gguf_name` is an identity function with strict allowlisting (rejects anything not matching a known global or per-block tensor pattern). This exists to catch checkpoint format drift early, not to do real renaming.
- `gguf_mapped.rs` — `MappedVarBuilder` wraps `quantized_var_builder::VarBuilder` and applies `gguf_mapping`'s name map when resolving tensors, falling back to the raw name if unmapped.
- `gguf_loader.rs` — standalone dequantize-GGUF-to-safetensors utility (used by `test_gguf.rs`), independent of the `MappedVarBuilder` quantized-load path used by `main.rs --quantized`.
- `rope.rs` — 3-axis (frame/height/width) RoPE (`EmbedNd`), matching diffusers' `QwenImage21Rope`: text tokens advance position on all three axes; each image block freezes the frame axis and lays out a height/width grid centered on zero. Supports `Precomputed` (default) or `Complex` frequency modes.
- `attention_mask.rs` — block-causal attention mask: a query may attend to any key at or before it (causal) OR any key in the same image block (`same_image_block AND image_id >= 0`). `build_token_metadata` derives per-token `image_ids`/`target_token_mask` from `img_shapes` — both transformer forward passes must pass the real `img_shapes` here (not `&[]`), otherwise every image token gets `image_id = -1` and the mask silently degenerates to pure causal, losing the bidirectional in-image-block attention the model needs. The checkpoint has `causal_condition: true`: `QwenImageTransformer::forward` gives text tokens a modulation row computed at t=0 and only image tokens the real timestep's row, so `TransformerBlock` receives a per-token `[B, seq, 4*hidden]` modulation (both transformers implement this).
- `scheduler.rs` — `FlowMatchEuler` (the checkpoint's dynamic exponential sigma shift + `shift_terminal` stretch, starting from `linspace(1, 1/N, N)` as the Qwen pipelines do; Euler step) and sinusoidal `timestep_embedding` (`[cos, sin]` order, i.e. `flip_sin_to_cos=True`).
- `vae.rs` — `VaeDecoder` implementing the real `AutoencoderKLQwenImage21` decoder (64-channel latents, `spatial_compression_ratio` = 16, RGBA/4-channel output, real `latents_mean`/`latents_std`). It's a causal-3D VAE (Wan-style) in diffusers, but for single-frame images `QwenImage21CausalConv3d` explicitly collapses to a plain `Conv2d`, so this module implements the decoder as pure 2D — no temporal/frame handling, no feature caching (both are only needed for multi-frame video decoding). Decoder-only: the encoder isn't implemented since the pipeline never encodes images. `dup_up3d()` reproduces the real depth-to-space upsample shortcut exactly (including the `factor_t` parameter, which still measurably changes the result on channel-reducing up-blocks even though the temporal axis itself is always squeezed back to size 1) — see `vae::tests::test_decoder_shapes` for the shape-wiring smoke test. Plus the latent `pack_latents`/`unpack_latents`/`normalize_latents` free functions the pipeline calls between transformer and VAE.
- `text_encoder.rs` — `TextEncoder::load` auto-detects the directory layout. **Official** (a `config.json` with `text_config`, e.g. the official repo's `text_encoder/` with its tokenizer in the sibling `processor/`): the real Qwen3-VL language model via `qwen3_vl_text.rs` (vendored because candle-transformers' `qwen3_vl` only exposes last-token logits). `encode` wraps the prompt in the upstream raw template (`<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`), takes the last decoder layer's output **before** the final RMSNorm, and drops the system-prefix tokens — all three are required to match what `txt_in` was trained on. **Stand-in** (a plain dense Qwen3 checkpoint, e.g. Qwen3-0.6B): hidden states tiled up to 4096 — prompt-dependent but semantically wrong; only useful when the official encoder doesn't fit.

### Known incomplete pieces (don't assume otherwise)


- Two independent GGUF-loading code paths exist: `gguf_loader.rs` (dequantize to F32 safetensors, used only by the standalone `test_gguf.rs` probe) vs. `gguf_mapped.rs`/`quantized_transformer.rs` (stay quantized, used by `main.rs --quantized`). Don't conflate them when touching GGUF loading.
- **`main.rs`'s `--quantized` CLI flag is dead**: `load_transformer` routes purely on whether `--model-path` ends in `.gguf`, never reads the `quantized` argument (rustc even flags it as unused).
- **`gguf_mapped::MappedVarBuilder`'s name mapping is bypassed in practice**: `main.rs` calls `mapped_vb.inner().clone()` before constructing `QwenImageTransformerQuantized`, discarding the wrapper. Harmless today because `gguf_mapping::map_gguf_name` is identity-only, but the mapping layer isn't actually exercised by any real code path.

### Models

`models/` holds real checkpoint files (GGUF and safetensors, multi-GB) plus a HuggingFace `.cache` download-lock directory — excluded via `.gitignore` (`/models`, alongside `/target`); don't force-add anything under it.
