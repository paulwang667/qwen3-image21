# Qwen-Image-2.1 Inference Engine (Rust)

A from-scratch Rust inference engine for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), built directly on `candle-core` / `candle-nn`. It runs the official full-precision weights or GGUF quantized weights, on CUDA, Metal, or CPU.

At 1024×1024 with 20 steps it produces images matching the reference implementation, e.g. for `"a red apple on a wooden table"`.

## Features

- **Single-stream MMDiT transformer**: 32 blocks, 32 heads × 128, with the checkpoint's `causal_condition` (text tokens use a t=0 modulation row, image tokens the real timestep's row).
- **Official Qwen3-VL text encoder**: text-only language model; prompt wrapped in the upstream template, last decoder state taken before the final norm, system-prefix tokens dropped.
- **Two transformer backends**:
  - official diffusers safetensors (sharded, F32 compute)
  - GGUF quantized weights, including ComfyUI-style GGUFs such as [`unsloth/Qwen-Image-2.1-GGUF`](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) (Q8_0 and Q4_K_M tested)
- **Text-prefix KV cache**: the text tokens' per-layer K/V are computed once and reused across denoising steps (on by default, as upstream).
- **FlowMatch Euler scheduler** with the checkpoint's resolution-dependent exponential shift and `shift_terminal`.
- **VAE decoder**: `AutoencoderKLQwenImage21`, 64-channel latents, 16× spatial compression, RGBA output (saved as RGB). Both the official diffusers VAE layout and the ComfyUI-repackaged layout load.
- **Optional true CFG** (`--negative-prompt` + `--true-cfg-scale`), off by default like upstream, since 2.1 is meant to be sampled without guidance.

## Getting the weights

From the official repo [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) you need `text_encoder/`, `processor/` (tokenizer), `vae/`, and, for full precision, `transformer/`:

```bash
hf download Qwen/Qwen-Image-2.1 --local-dir models/Qwen-Image-2.1-official
```

For the quantized path, add a GGUF transformer:

```bash
hf download unsloth/Qwen-Image-2.1-GGUF qwen-image-2.1-Q4_K_M.gguf --local-dir models
```

The GGUF replaces only the transformer; the text encoder and VAE still come from the official repo.

## Build

```bash
cargo build --release --features cuda    # Linux + NVIDIA
cargo build --release --features metal   # macOS / Apple Silicon
cargo build --release                    # CPU only
```

At runtime the binary tries CUDA, then Metal, then falls back to CPU.

## Usage

Quantized (GGUF) transformer:

```bash
R=models/Qwen-Image-2.1-official
./target/release/qwen3-image21 \
  --prompt "a red apple on a wooden table" \
  --model-path models/qwen-image-2.1-Q4_K_M.gguf \
  --vae-path $R/vae/diffusion_pytorch_model.safetensors \
  --text-encoder-path $R/text_encoder \
  --height 1024 --width 1024 --steps 20 \
  --output apple.png
```

Full precision: point `--model-path` at the first transformer shard. The remaining shards are found through the `*.safetensors.index.json` next to it.

```bash
  --model-path $R/transformer/diffusion_pytorch_model-00001-of-00002.safetensors
```

With classifier-free guidance (roughly doubles the denoising time):

```bash
  --negative-prompt "blurry, low quality" --true-cfg-scale 4
```

### Options

| Option | Default | Notes |
|---|---|---|
| `--prompt` | (required) | |
| `--output` | `output.png` | |
| `--height`, `--width` | `1024` | Multiples of 16 (one latent cell = 16×16 px). |
| `--steps` | `40` | Upstream default. 20 already gives clean results. |
| `--model-path` | (required) | `.gguf` → quantized path; otherwise safetensors. |
| `--vae-path` | | |
| `--text-encoder-path` | | The official `text_encoder/` directory (a dense Qwen3 stand-in also loads, but gives semantically wrong conditioning). Omitting it uses random embeddings. |
| `--negative-prompt` | | Only used when `--true-cfg-scale > 1`. |
| `--true-cfg-scale` | `1.0` (off) | `neg + scale × (cond − neg)`, no rescaling, as upstream. |
| `--no-kv-cache` | off | Recompute the text prefix every step (for A/B checks). |
| `--precision` | `f32` | Only `f32` has been verified end to end. |
| `--benchmark`, `--benchmark-iterations` | off, `3` | Keeps transformer + VAE loaded and times repeated runs. |
| `--seed` | `42` | **Currently ignored**; noise comes from candle's default RNG. |
| `--quantized` | | **Currently ignored**; the backend is chosen by the `--model-path` extension. |

## Performance

NVIDIA L20 (46 GB), 1024×1024, 20 steps, denoising time only:

| Transformer | Size on disk | KV cache on | KV cache off |
|---|---|---|---|
| Full precision (F32 compute) | 14 GB (BF16) | 89.9 s | 99.2 s |
| Q8_0 GGUF | 7.6 GB | 58.0 s | 67.1 s |
| Q4_K_M GGUF | 4.2 GB | 58.3 s | 67.3 s |

- Q4_K_M and Q8_0 run at the same speed: candle's CUDA quantized matmul appears to dequantize for sequences this long, so fewer bits mainly save disk and memory.
- The KV-cache gain comes mostly from dropping the per-layer block-causal attention mask on cached steps, not from skipping the ~15 text tokens.
- Accuracy vs. full precision (one forward pass, cosine similarity): Q8_0 0.99986, Q4_K_M 0.99537. Neither shows a visible quality difference.

Models are loaded in phases (text encoder → transformer → VAE), each dropped before the next. Everything is loaded as F32, so peak memory is the text-encoder phase: about 34 GB of F32 weights, estimated from the 17 GB BF16 checkpoint.

## Project layout

```
src/
├── main.rs                  # CLI, phased model loading, image saving
├── pipeline.rs              # denoising loop (CFG, KV cache), latent decoding
├── transformer.rs           # full-precision transformer, TextKvCache
├── quantized_transformer.rs # GGUF transformer (same math, quantized matmuls)
├── text_encoder.rs          # prompt template + encoder backends
├── qwen3_vl_text.rs         # vendored Qwen3-VL text model (pre-norm hidden states)
├── rope.rs                  # 3-axis (frame/height/width) RoPE
├── attention_mask.rs        # block-causal mask (causal text, bidirectional image)
├── scheduler.rs             # FlowMatch Euler + sinusoidal timestep embedding
├── vae.rs                   # AutoencoderKLQwenImage21 decoder
└── bin/                     # diagnostics (see below)
```

## Diagnostics

The `src/bin/` tools were used to find and verify the fixes that made this engine match the reference:

| Binary | Checks |
|---|---|
| `tensor_coverage_probe` | Every checkpoint tensor is consumed by the model code (no silently dropped weights). |
| `gguf_weight_probe` | Dequantized GGUF tensors match the official safetensors. |
| `quant_vs_full_probe` | One forward pass: full precision vs. quantized on GPU vs. quantized on CPU. |
| `kv_cache_probe` | Cached and uncached forwards agree. |
| `text_conditioning_probe` | Different prompts change the transformer's prediction. |
| `real_latent_probe`, `vae_probe`, `synth_latents` | Decode real or synthetic latents directly through the VAE. |

```bash
cargo test    # unit tests (RoPE vs. reference formula, dup_up3d, scheduler, attention mask, …)
```

## Limitations

- Text-to-image only; image-conditioned editing (the model's condition-image slots) is not implemented.
- `--seed` has no effect yet, and `--quantized` is redundant.
- Only F32 compute has been verified; `--precision f16/bf16` is untested.
- End-to-end quality has been verified on CUDA only. The CPU path was checked at the single-forward level (quantized on CPU matches full precision); Metal has not been re-verified since the correctness fixes.
- No flash attention: attention scores are materialised in full.

## License

MIT
