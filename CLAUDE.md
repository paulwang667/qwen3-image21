# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch Rust inference engine for Qwen-Image-2.1 (text-to-image MMDiT diffusion model), built directly on `candle-core`/`candle-nn` (not `candle-transformers`' model zoo — only its `quantized_nn`/`quantized_var_builder` utilities are reused for GGUF loading). Targets Apple Silicon via the Metal backend.

The model weights the code is written against use a **single-stream** MMDiT architecture (one sequence of concatenated text+image tokens per block, shared modulation across all blocks) — not the classic Flux-style dual-stream (separate `img_attn`/`txt_attn`) design. `list_tensors.rs` and `test_gguf.rs` at the repo root are leftover scratch probes from reverse-engineering the checkpoint's tensor names; they are **not** part of the cargo build (no `[[bin]]` entries reference them) and can be ignored or deleted.

## Commands

```bash
cargo check                    # fast type-check, use this while iterating
cargo build --release          # release build (needed for real inference speed)
cargo test                     # run unit tests (gguf_mapping, attention_mask)
cargo test test_identity_global   # run a single test by name

# Run inference (requires a real transformer + VAE model on disk; see models/)
cargo run --release -- --prompt "a cat" --model-path models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors --vae-path models/vae/qwen_image_2.1_vae_bf16.safetensors
cargo run --release -- --prompt "a cat" --model-path models/qwen-image-2.1-Q4_0.gguf --quantized
cargo run --release -- --prompt "a cat" --model-path <model> --benchmark --benchmark-iterations 5

# With the stand-in text encoder (a directory with config.json/model.safetensors/tokenizer.json
# for a standard dense Qwen3 model — see text_encoder.rs); omit for random-embedding fallback
cargo run --release -- --prompt "a cat" --model-path models/qwen-image-2.1-Q4_0.gguf --quantized --text-encoder-path <qwen3-model-dir>

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
- `transformer.rs` — full-precision `QwenImageTransformer` (`VarBuilder`-based, safetensors). Defines the single-stream `TransformerBlock` (LayerNorm → shared modulation split into scale/gate pairs → attention → MLP), `Attention` (Q/K RMSNorm + RoPE), `GatedMlp` (SwiGLU: `out(silu(proj(x)) * gate_layer(x))`), and `TextEmbedder` (ZeroCenterRMSNorm text projection — checkpoint stores `scale - 1`, code adds 1 back). `Config::default()` describes the actual checkpoint shape (32 layers, 32 heads, head_dim 128 → hidden 4096), matching README.md's Model Requirements section.
- `quantized_transformer.rs` — mirrors `transformer.rs` block-for-block but built on `candle_transformers::quantized_nn`/`quantized_var_builder` so weights stay quantized (no dequantize-to-F32 step) during the forward pass.
- `gguf_mapping.rs` — pure name-validation: the checkpoint's GGUF tensor names already match this code's expected names 1:1, so `map_gguf_name` is an identity function with strict allowlisting (rejects anything not matching a known global or per-block tensor pattern). This exists to catch checkpoint format drift early, not to do real renaming.
- `gguf_mapped.rs` — `MappedVarBuilder` wraps `quantized_var_builder::VarBuilder` and applies `gguf_mapping`'s name map when resolving tensors, falling back to the raw name if unmapped.
- `gguf_loader.rs` — standalone dequantize-GGUF-to-safetensors utility (used by `test_gguf.rs`), independent of the `MappedVarBuilder` quantized-load path used by `main.rs --quantized`.
- `rope.rs` — 3-axis (frame/height/width) RoPE (`EmbedNd`), matching diffusers' `QwenImage21Rope`: text tokens advance position on all three axes; each image block freezes the frame axis and lays out a height/width grid centered on zero. Supports `Precomputed` (default) or `Complex` frequency modes.
- `attention_mask.rs` — block-causal attention mask: a query may attend to any key at or before it (causal) OR any key in the same image block (`same_image_block AND image_id >= 0`). `build_token_metadata` derives per-token `image_ids`/`target_token_mask` from `img_shapes` — both transformer forward passes must pass the real `img_shapes` here (not `&[]`), otherwise every image token gets `image_id = -1` and the mask silently degenerates to pure causal, losing the bidirectional in-image-block attention the model needs. `select_modulation_rows` only implements the plain (non-`causal_condition`) case — real diffusers' `[batch+1, dim]` trailing-`t=0`-row split for condition-image/editing generation isn't implemented, since nothing in this codebase ever constructs that extra row.
- `scheduler.rs` — `FlowMatchEuler` (linear sigma schedule, Euler step) and sinusoidal `timestep_embedding`.
- `vae.rs` — `VaeDecoder` implementing the real `AutoencoderKLQwenImage21` decoder (64-channel latents, `spatial_compression_ratio` = 16, RGBA/4-channel output, real `latents_mean`/`latents_std`). It's a causal-3D VAE (Wan-style) in diffusers, but for single-frame images `QwenImage21CausalConv3d` explicitly collapses to a plain `Conv2d`, so this module implements the decoder as pure 2D — no temporal/frame handling, no feature caching (both are only needed for multi-frame video decoding). Decoder-only: the encoder isn't implemented since the pipeline never encodes images. `dup_up3d()` reproduces the real depth-to-space upsample shortcut exactly (including the `factor_t` parameter, which still measurably changes the result on channel-reducing up-blocks even though the temporal axis itself is always squeezed back to size 1) — see `vae::tests::test_decoder_shapes` for the shape-wiring smoke test. Plus the latent `pack_latents`/`unpack_latents`/`normalize_latents` free functions the pipeline calls between transformer and VAE.
- `text_encoder.rs` — **not the real text encoder**: the actual Qwen-Image-2.1 pipeline uses a hybrid Qwen3-Next-style VLM (linear-attention/Mamba2 layers periodically mixed with regular attention — see the real checkpoint's `linear_attn.{A_log,dt_bias,conv1d,in_proj_*}` tensor names) with a non-standard int8 "rotation" quantization scheme (`weight`/`weight_scale`/a 72-byte `comfy_quant` blob per tensor). candle-transformers has no implementation of either the architecture or the quantization format, and building one from scratch was judged out of scope. Instead this loads a real, standard dense `candle_transformers::models::qwen3::Model` (e.g. a local Qwen3-0.6B checkpoint — hidden_size 1024) via `TextEncoder::load(model_dir, joint_attention_dim, device)`, and tiles its hidden states up to the transformer's `joint_attention_dim` (4096 = 1024 × 4) so they still fit through the checkpoint's real, trained `txt_in` layer. This gives genuine, deterministic, prompt-dependent conditioning — confirmed different prompts produce different embeddings and the same prompt is reproducible — but the semantics don't match what `txt_in` was actually trained on, so don't expect coherent images from it alone.

### Known incomplete pieces (don't assume otherwise)

- **Text encoder is a dimensionality-matched stand-in, not the real model** (see `text_encoder.rs` above) — expect prompts to have *some* influence, not the real model's semantics.
- **KV-cache exists but is wired to nothing.** `transformer.rs`/`quantized_transformer.rs` both implement `forward_cached`/`KVCache`, but `pipeline::denoise` always calls the uncached `forward`, and each denoising step recomputes the full sequence.
- Two independent GGUF-loading code paths exist: `gguf_loader.rs` (dequantize to F32 safetensors, used only by the standalone `test_gguf.rs` probe) vs. `gguf_mapped.rs`/`quantized_transformer.rs` (stay quantized, used by `main.rs --quantized`). Don't conflate them when touching GGUF loading.
- **`main.rs`'s `--quantized` CLI flag is dead**: `load_transformer` routes purely on whether `--model-path` ends in `.gguf`, never reads the `quantized` argument (rustc even flags it as unused).
- **`gguf_mapped::MappedVarBuilder`'s name mapping is bypassed in practice**: `main.rs` calls `mapped_vb.inner().clone()` before constructing `QwenImageTransformerQuantized`, discarding the wrapper. Harmless today because `gguf_mapping::map_gguf_name` is identity-only, but the mapping layer isn't actually exercised by any real code path.
- **No resolution-dependent sigma shift**: `scheduler::FlowMatchEuler::calculate_shift` exists but nothing calls it; `pipeline::denoise` always uses the plain unshifted linear schedule `sigma_i = 1 - i/N`, unlike diffusers' `FlowMatchEulerDiscreteScheduler` which shifts sigmas based on image sequence length.

### Models

`models/` holds real checkpoint files (GGUF and safetensors, multi-GB) plus a HuggingFace `.cache` download-lock directory — excluded via `.gitignore` (`/models`, alongside `/target`); don't force-add anything under it.
