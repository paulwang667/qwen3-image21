"""Dump golden intermediate tensors from the upstream diffusers QwenImage21Pipeline
for image-conditioned generation, so each stage of the Rust port can be checked
numerically. Everything runs in float32 to match the Rust F32 path.

Usage: python tools/ref_dump_i2i.py <model_dir> <condition_image> <out_dir> [resolution] [prompt]
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from diffusers import QwenImage21Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

model_dir, image_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
res = int(sys.argv[4]) if len(sys.argv) > 4 else 512
prompt = sys.argv[5] if len(sys.argv) > 5 else "Change the apple to a green apple"
os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda")
# Full-precision reference: PyTorch runs cuDNN convolutions in TF32 by default,
# which alone puts ~1e-3 relative noise into the VAE and vision stages.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def save(name, tensors):
    tensors = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
    save_file(tensors, os.path.join(out_dir, f"{name}.safetensors"))
    print(f"saved {name}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in tensors.items()), flush=True)


pipe = QwenImage21Pipeline.from_pretrained(model_dir, torch_dtype=torch.float32)

# --- 1. condition image preprocessing (exactly as __call__ does it) ---
img = Image.open(image_path).convert("RGBA")
w, h, _ = calculate_dimensions(res * res, img.size[0] / img.size[1])
resized = pipe.image_processor.resize(img, width=w, height=h)
vae_image = pipe.image_processor.preprocess(img, width=w, height=h).unsqueeze(2)  # [1,4,1,H,W] in [-1,1]
save("image", {"vae_image": vae_image, "resized_rgba": torch.from_numpy(np.array(resized))})

# --- 2. text encoder (with vision), capturing processor outputs and vision features ---
pipe.text_encoder.to(dev)
captured = {}
orig_forward = pipe.text_encoder.forward


def capture_forward(*args, **kwargs):
    for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"):
        if kwargs.get(k) is not None:
            captured[k] = kwargs[k]
    return orig_forward(*args, **kwargs)


pipe.text_encoder.forward = capture_forward
visual = pipe.text_encoder.model.visual


def vision_hook(module, args, output):
    # Qwen3-VL's visual tower returns (image_embeds, deepstack_feature_list) or a model output.
    if isinstance(output, tuple):
        captured["image_embeds"] = output[0]
        for i, t in enumerate(output[1]):
            captured[f"deepstack_{i}"] = t
    else:
        captured["image_embeds"] = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state
        for i, t in enumerate(getattr(output, "deepstack_features", None) or []):
            captured[f"deepstack_{i}"] = t


vh = visual.register_forward_hook(vision_hook)
with torch.no_grad():
    prompt_embeds, prompt_mask, image_pad_mask = pipe.encode_prompt(prompt=prompt, image=[resized], device=dev)
vh.remove()
pipe.text_encoder.forward = orig_forward
save("processor", {k: captured[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw") if k in captured})
save("vision", {k: v for k, v in captured.items() if k == "image_embeds" or k.startswith("deepstack_")})
save("text", {"prompt_embeds": prompt_embeds, "image_pad_mask": image_pad_mask.to(torch.uint8)})
pipe.text_encoder.to("cpu")
torch.cuda.empty_cache()

# --- 3. VAE encode ---
pipe.vae.to(dev)
with torch.no_grad():
    image_latents = pipe._encode_vae_image(vae_image.to(dev), None)  # [1,64,1,h,w], normalized
save("vae", {"image_latents": image_latents})
pipe.vae.to("cpu")

# --- 4. one transformer forward on fixed inputs (no KV cache) ---
pipe.transformer.to(dev)
th, tw = h // 16, w // 16  # target same size as the condition here
g = torch.Generator(device="cpu").manual_seed(0)
noise = torch.randn((1, th * tw, 64), generator=g).to(dev)
cond_tokens = image_latents.view(1, 64, -1).transpose(1, 2)
img_shapes = [[(1, h // 16, w // 16), (1, th, tw)]]
img_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(1, th * tw // 4)], dim=1)
sigma = 0.6
with torch.no_grad():
    noise_pred = pipe.transformer(
        hidden_states=torch.cat([cond_tokens, noise], dim=1),
        timestep=torch.tensor([sigma], device=dev),
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
    )[0]
save("transformer", {"noise": noise, "sigma": torch.tensor([sigma]), "noise_pred": noise_pred[:, -th * tw :]})
pipe.transformer.to("cpu")
torch.cuda.empty_cache()

if os.environ.get("SKIP_E2E"):
    sys.exit(0)

# --- 5. end-to-end reference image, from known initial noise, recording every step ---
steps = int(os.environ.get("STEPS", "20"))
init = torch.randn((1, th * tw, 64), generator=torch.Generator(device="cpu").manual_seed(1))
per_step = {}


def record(pipeline, i, t, kwargs):
    per_step[f"step_{i}"] = kwargs["latents"].detach().float().cpu()
    return kwargs


pipe.enable_model_cpu_offload()
out = pipe(
    prompt=prompt,
    image=[img],
    num_inference_steps=steps,
    output_resolution=res,
    latents=init.to(dev),
    callback_on_step_end=record,
    callback_on_step_end_tensor_inputs=["latents"],
)
out.images[0].save(os.path.join(out_dir, "reference.png"))
save("loop", {"init": init, "sigmas": pipe.scheduler.sigmas.float(), **per_step})
print("saved reference.png", flush=True)
