# Qwen-Image-2.1 Inference Engine (Rust)

A from-scratch Rust inference engine for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), built directly on `candle-core` / `candle-nn`. It runs the official full-precision weights or GGUF quantized weights, on CUDA, Metal, or CPU, for both **text-to-image** and **image-conditioned generation** (editing / reference images).

Each stage (preprocessing, vision tower, text encoder, VAE, transformer, and the full denoising loop) has been checked numerically against the upstream diffusers pipeline and matches to ~1e-5.

## Features

- **Single-stream MMDiT transformer**: 32 blocks, 32 heads × 128, with the checkpoint's `causal_condition` (text tokens use a t=0 modulation row, image tokens the real timestep's row).
- **Official Qwen3-VL text encoder**: language model plus vision tower (for condition images, with multimodal RoPE and DeepStack); prompt wrapped in the upstream template, last decoder state taken before the final norm, system-prefix tokens dropped.
- **Two transformer backends**:
  - official diffusers safetensors (sharded, F32 compute)
  - GGUF quantized weights, including ComfyUI-style GGUFs such as [`unsloth/Qwen-Image-2.1-GGUF`](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) (Q8_0 and Q4_K_M tested)
- **Prefix KV cache**: the per-layer K/V of the text and condition-image tokens are computed once and reused across denoising steps (on by default, as upstream).
- **FlowMatch Euler scheduler** with the checkpoint's resolution-dependent exponential shift and `shift_terminal`.
- **VAE**: `AutoencoderKLQwenImage21` decoder and encoder, 64-channel latents, 16× spatial compression, RGBA (output saved as RGB). The decoder loads both the official diffusers layout and the ComfyUI-repackaged layout; the encoder needs the official one.
- **Image-conditioned generation** (`--image`, repeatable): the condition image is read by the Qwen3-VL vision tower as part of the prompt and VAE-encoded into latent tokens placed in the prompt's image slots, as in upstream `QwenImage21Pipeline`. Works for edits ("change the apple to a green apple"), style changes ("turn this photo into a watercolor painting"), and object replacement.
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

Image-conditioned generation (editing): pass one or more `--image`. The output size defaults to the last condition image's aspect ratio at `--output-resolution` (1024² area):

```bash
./target/release/qwen3-image21 \
  --prompt "Change the apple to a green apple" \
  --image apple.png \
  --model-path models/qwen-image-2.1-Q4_K_M.gguf \
  --vae-path $R/vae/diffusion_pytorch_model.safetensors \
  --text-encoder-path $R/text_encoder \
  --steps 20 --output green_apple.png
```

Don't edit an image with the seed it was generated from: identical starting noise locks the edit onto the original and produces an oversaturated copy. The default random seed avoids this.

With classifier-free guidance (roughly doubles the denoising time):

```bash
  --negative-prompt "blurry, low quality" --true-cfg-scale 4
```

### Options

| Option | Default | Notes |
|---|---|---|
| `--prompt` | (required) | |
| `--output` | `output.png` | |
| `--height`, `--width` | `1024` | Multiples of 16 (one latent cell = 16×16 px). With `--image`, default to the condition image's size. |
| `--image` | | Condition image (repeatable). Needs the official text encoder (it contains the vision tower). |
| `--output-resolution` | `1024` | Condition images are resized to about this² pixels at their own aspect ratio (multiples of 32). |
| `--steps` | `40` | Upstream default. 20 already gives clean results. |
| `--model-path` | (required) | `.gguf` → quantized path; otherwise safetensors. |
| `--vae-path` | | |
| `--text-encoder-path` | | The official `text_encoder/` directory (a dense Qwen3 stand-in also loads, but gives semantically wrong conditioning). Omitting it uses random embeddings. |
| `--negative-prompt` | | Only used when `--true-cfg-scale > 1`. |
| `--true-cfg-scale` | `1.0` (off) | `neg + scale × (cond − neg)`, no rescaling, as upstream. |
| `--no-kv-cache` | off | Recompute the text prefix every step (for A/B checks). |
| `--precision` | `f32` | Only `f32` has been verified end to end. |
| `--benchmark`, `--benchmark-iterations` | off, `3` | Keeps transformer + VAE loaded and times repeated runs. |
| `--seed` | random | Printed at startup; pass it again to reproduce a run. Noise is drawn on the host, so a seed gives the same noise on every device. |
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

Image-conditioned generation at 1024×1024 (one 1024² condition image, 20 steps) roughly doubles the sequence to ~8,200 tokens: full precision 135 s (peak ~44 GB on the 46 GB L20), Q4_K_M 102 s. Attention is computed in chunks of 1,024 query rows so the score matrix never has to fit at once.

Models are loaded in phases (text encoder → VAE encoder for condition images → transformer → VAE decoder), each dropped before the next. Everything is loaded as F32. For text-to-image, peak memory is the text-encoder phase (about 34 GB of F32 weights, estimated from the 17 GB BF16 checkpoint); image-conditioned generation in full precision peaks in the transformer phase (~44 GB measured at 1024²), so use a GGUF transformer on smaller GPUs.

## Project layout

```
src/
├── main.rs                  # CLI, phased model loading, image saving
├── pipeline.rs              # denoising loop (CFG, KV cache), latent decoding
├── transformer.rs           # full-precision transformer, TextKvCache
├── quantized_transformer.rs # GGUF transformer (same math, quantized matmuls)
├── text_encoder.rs          # prompt templates (text / with images) + encoder backends
├── qwen3_vl_text.rs         # Qwen3-VL language model (multimodal RoPE, DeepStack, pre-norm states)
├── qwen3_vl_vision.rs       # Qwen3-VL vision tower (vendored from candle-transformers)
├── condition_image.rs       # condition-image resize + VAE/vision preprocessing
├── joint_layout.rs          # transformer sequence: text, condition-image blocks, target
├── rope.rs                  # 3-axis (frame/height/width) RoPE
├── attention_mask.rs        # block-causal mask (causal text, bidirectional image)
├── scheduler.rs             # FlowMatch Euler + sinusoidal timestep embedding
├── vae.rs                   # AutoencoderKLQwenImage21 encoder + decoder
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
| `i2i_probe` | Each stage of image-conditioned generation (`vae`, `pre`, `vision`, `text`, `transformer`, `encode`, `loop`) against golden tensors from `tools/ref_dump_i2i.py`, which runs the upstream diffusers pipeline (needs torch + diffusers main + transformers). |
| `text_conditioning_probe` | Different prompts change the transformer's prediction. |
| `real_latent_probe`, `vae_probe`, `synth_latents` | Decode real or synthetic latents directly through the VAE. |

```bash
cargo test    # unit tests (RoPE vs. reference formula, dup_up3d, scheduler, attention mask, …)
```

## Limitations

- `--quantized` is redundant (the backend is chosen by the `--model-path` extension).
- Condition images are resized with the `image` crate's Lanczos3, close to but not bit-identical with upstream's PIL resize.
- Only F32 compute has been verified; `--precision f16/bf16` is untested.
- End-to-end quality has been verified on CUDA only. The CPU path was checked at the single-forward level (quantized on CPU matches full precision); Metal has not been re-verified since the correctness fixes.
- No flash attention: attention scores are materialised in full.

## License

MIT
