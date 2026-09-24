#!/usr/bin/env bash
# Downloads the weights into the layout `--model-dir` expects:
#
#   <dir>/qwen-image-2.1-<quant>.gguf        (q4 / q8)
#   <dir>/Qwen-Image-2.1-official/{text_encoder,processor,vae}/
#                                 transformer/    (full only)
#
# Usage: scripts/download.sh [q4|q8|full] [dir]     (defaults: q4, models)
#   q4    Q4_K_M GGUF transformer (4.2 GB) — fits a 12 GB GPU; ~23 GB in total
#   q8    Q8_0 GGUF transformer (7.6 GB); ~27 GB in total
#   full  official full-precision transformer (14 GB); ~33 GB in total
# Plus, for every variant, the official text encoder (17.5 GB), tokenizer, and
# VAE (1.4 GB). Needs the `hf` CLI (pip install -U huggingface_hub); set
# HF_ENDPOINT for a mirror (e.g. https://hf-mirror.com). Re-running resumes.
# DRY_RUN=1 lists the files without downloading them.
set -euo pipefail

variant="${1:-q4}"
dir="${2:-models}"
repo_dir="$dir/Qwen-Image-2.1-official"

case "$variant" in
  q4) gguf="qwen-image-2.1-Q4_K_M.gguf" ;;
  q8) gguf="qwen-image-2.1-Q8_0.gguf" ;;
  full) gguf="" ;;
  *) echo "usage: $0 [q4|q8|full] [dir]" >&2; exit 2 ;;
esac

if ! command -v hf >/dev/null 2>&1; then
  echo "error: the 'hf' CLI was not found; install it with: pip install -U huggingface_hub" >&2
  exit 1
fi

# A plain word, not an array: bash 3.2 (macOS) rejects empty arrays under `set -u`.
dry_run=""
if [[ "${DRY_RUN:-}" == 1 ]]; then dry_run="--dry-run"; fi

# Every pattern needs its own --include: values after one --include are read as file names.
includes=(--include "text_encoder/*" --include "processor/*" --include "vae/*")
if [[ -n "$gguf" ]]; then
  echo "== GGUF transformer: $gguf"
  hf download unsloth/Qwen-Image-2.1-GGUF "$gguf" --local-dir "$dir" $dry_run
  parts="text encoder, tokenizer, VAE"
else
  includes+=(--include "transformer/*")
  parts="text encoder, tokenizer, VAE, transformer"
fi

echo "== Official repo ($parts) -> $repo_dir"
hf download Qwen/Qwen-Image-2.1 "${includes[@]}" --local-dir "$repo_dir" $dry_run

if [[ -n "$dry_run" ]]; then exit 0; fi

echo
ggufs=("$dir"/*.gguf)
if [[ -n "$gguf" && ${#ggufs[@]} -gt 1 ]]; then
  echo "Note: $dir holds several GGUF files; pick one with --model-path $dir/$gguf"
elif [[ -z "$gguf" && -e "${ggufs[0]}" ]]; then
  echo "Note: $dir also holds a GGUF file, which is preferred; for full precision pass"
  echo "  --model-path $repo_dir/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
fi
model_dir_arg=""
if [[ "$dir" != "models" ]]; then model_dir_arg=" --model-dir $dir"; fi
echo "Done. Run:"
echo "  ./target/release/qwen3-image21 --prompt \"a red apple on a wooden table\" --steps 20$model_dir_arg"
