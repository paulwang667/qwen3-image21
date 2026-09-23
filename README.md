# Qwen-Image-2.1 Quantized Inference Engine

A Rust-based inference engine for Qwen-Image-2.1 text-to-image model, built on candle-transformers.

## Features

- **MMDiT Architecture**: Implements the Multi-Modal Diffusion Transformer with dual streams (image + text)
- **Quantized Inference**: Support for GGUF quantized weights via candle-transformers quantized_nn
- **RoPE Embeddings**: Rotary Position Embeddings for 2D image patches
- **FlowMatchEuler Scheduler**: Euler method for flow matching diffusion
- **VAE Decoder**: AutoencoderKL for latent space decoding
- **Benchmark Mode**: Measure inference performance with multiple iterations
- **Metal Backend**: Optimized for Apple Silicon (Metal GPU)

## Architecture

```
src/
├── main.rs              # CLI entry point with benchmark support
├── lib.rs               # Module declarations
├── transformer.rs       # Non-quantized MMDiT transformer
├── quantized_transformer.rs  # Quantized variant using GGUF
├── vae.rs               # VAE decoder
├── pipeline.rs          # Complete inference pipeline
├── scheduler.rs         # FlowMatchEulerDiscreteScheduler
└── rope.rs              # RoPE embeddings
```

## Usage

### Basic Usage

```bash
cargo run --release -- \
  --prompt "A beautiful sunset over the ocean" \
  --model-path path/to/model.gguf \
  --output output.png \
  --height 1024 \
  --width 1024 \
  --steps 50
```

### Quantized Mode

```bash
cargo run --release -- \
  --prompt "A cat sitting on a table" \
  --model-path path/to/model-quantized.gguf \
  --quantized \
  --output output.png
```

### Benchmark Mode

```bash
cargo run --release -- \
  --prompt "A dog playing in the park" \
  --model-path path/to/model.gguf \
  --benchmark \
  --benchmark-iterations 5
```

### CLI Options

```
Options:
      --prompt <PROMPT>                              Prompt for image generation
      --output <OUTPUT>                              Output image file [default: output.png]
      --height <HEIGHT>                              Image height [default: 1024]
      --width <WIDTH>                                Image width [default: 1024]
      --steps <STEPS>                                Number of inference steps [default: 50]
      --quantized                                    Use quantized model
      --benchmark                                    Run benchmark mode (multiple iterations)
      --benchmark-iterations <BENCHMARK_ITERATIONS>  Number of benchmark iterations [default: 3]
      --model-path <MODEL_PATH>                      Model path (GGUF or safetensors)
      --vae-path <VAE_PATH>                          VAE model path
      --text-encoder-path <TEXT_ENCODER_PATH>        Text encoder model path
      --seed <SEED>                                  Seed for random generation [default: 42]
  -h, --help                                         Print help
  -V, --version                                      Print version
```

## Model Requirements

### Transformer Model

- Architecture: MMDiT (Multi-Modal Diffusion Transformer)
- Format: GGUF (quantized) or safetensors (non-quantized)
- Layers: 32 transformer blocks
- Attention heads: 32
- Hidden size: 4096

### VAE Model

- Architecture: AutoencoderKL
- Format: safetensors
- Latent channels: 16
- Spatial compression: 8x

### Text Encoder (Placeholder)

- Architecture: Qwen2.5-VL
- Format: safetensors (planned)
- Output dimension: 3584

## Development

### Build

```bash
cargo build --release
```

### Check

```bash
cargo check
```

### Run

```bash
cargo run --release -- --prompt "test" --model-path model.gguf
```

## Limitations

- Text encoder is currently a placeholder (generates random embeddings)
- No flash-attention optimization yet
- No KV-cache support for iterative denoising
- INT8/INT4 quantization not yet implemented (GGUF only)

## Roadmap

- [ ] Implement Qwen2.5-VL text encoder integration
- [ ] Add INT8/INT4 quantization support
- [ ] Optimize attention with flash-attention
- [ ] Add KV-cache support
- [ ] Add comprehensive tests
- [ ] Compare output quality with reference implementation

## License

MIT
