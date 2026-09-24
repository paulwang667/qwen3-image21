//! Diagnostic: load each model through a recording VarBuilder backend and diff
//! the tensor names our code actually requests against every tensor stored in
//! the checkpoint. A checkpoint tensor that is never requested (e.g. a bias the
//! code builds its Linear without) is silently dropped weight — exactly the kind
//! of mismatch that loads without error yet corrupts the forward pass.
use anyhow::Result;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

struct Recording {
    inner: MmapedSafetensors,
    requested: Arc<Mutex<BTreeSet<String>>>,
}

impl SimpleBackend for Recording {
    fn get(&self, s: Shape, name: &str, h: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.requested.lock().unwrap().insert(name.to_string());
        SimpleBackend::get(&self.inner, s, name, h, dtype, dev)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.requested.lock().unwrap().insert(name.to_string());
        SimpleBackend::get_unchecked(&self.inner, name, dtype, dev)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

fn recording_vb(paths: &[String], device: &Device) -> Result<(VarBuilder<'static>, Arc<Mutex<BTreeSet<String>>>, Vec<(String, Vec<usize>)>)> {
    let inner = unsafe { MmapedSafetensors::multi(paths)? };
    let all: Vec<(String, Vec<usize>)> = inner.tensors().into_iter().map(|(n, v)| (n, v.shape().to_vec())).collect();
    let requested = Arc::new(Mutex::new(BTreeSet::new()));
    let backend = Recording { inner, requested: requested.clone() };
    Ok((VarBuilder::from_backend(Box::new(backend), DType::F32, device.clone()), requested, all))
}

fn report(label: &str, all: &[(String, Vec<usize>)], requested: &BTreeSet<String>, keep: impl Fn(&str) -> bool) {
    let stored: BTreeSet<&str> = all.iter().map(|(n, _)| n.as_str()).filter(|n| keep(n)).collect();
    let unused: Vec<&(String, Vec<usize>)> = all.iter().filter(|(n, _)| keep(n) && !requested.contains(n)).collect();
    let missing: Vec<&String> = requested.iter().filter(|n| !stored.contains(n.as_str())).collect();
    println!("\n=== {label} ===");
    println!("stored (in scope): {}  requested: {}  unused: {}  requested-but-absent: {}", stored.len(), requested.len(), unused.len(), missing.len());
    // Collapse per-block names ("transformer_blocks.17.attn.to_q.bias" ->
    // "transformer_blocks.*.attn.to_q.bias") so a systematic miss prints once.
    let mut patterns: std::collections::BTreeMap<String, (usize, Vec<usize>)> = Default::default();
    for (n, shape) in &unused {
        let pat: Vec<String> = n.split('.').map(|p| if p.parse::<usize>().is_ok() { "*".to_string() } else { p.to_string() }).collect();
        let e = patterns.entry(pat.join(".")).or_insert((0, shape.clone()));
        e.0 += 1;
    }
    for (pat, (count, shape)) in &patterns {
        println!("  UNUSED x{count:<4} {pat}  {shape:?}");
    }
    for n in &missing {
        println!("  ABSENT        {n}");
    }
}

fn main() -> Result<()> {
    let device = match Device::cuda_if_available(0)? {
        Device::Cpu => Device::metal_if_available(0)?,
        d => d,
    };
    let root = std::env::args().nth(1).unwrap_or_else(|| "models/Qwen-Image-2.1-official".to_string());

    // Transformer
    {
        let paths = qwen3_image21::safetensors_util::resolve_safetensors_paths(&format!(
            "{root}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
        ))?;
        let (vb, requested, all) = recording_vb(&paths, &device)?;
        let cfg = qwen3_image21::transformer::Config::default();
        let model = qwen3_image21::transformer::QwenImageTransformer::new(&cfg, vb)?;
        drop(model);
        report("transformer", &all, &requested.lock().unwrap(), |_| true);
    }

    // VAE (decoder only by design — encoder.* is expected to be unused)
    {
        let paths = vec![format!("{root}/vae/diffusion_pytorch_model.safetensors")];
        let (vb, requested, all) = recording_vb(&paths, &device)?;
        let model = qwen3_image21::vae::VaeDecoder::new(&qwen3_image21::vae::Config::default(), vb)?;
        drop(model);
        report("vae (excluding encoder.* / quant_conv.*)", &all, &requested.lock().unwrap(), |n| {
            !n.starts_with("encoder.") && !n.starts_with("quant_conv.")
        });
    }

    // Text encoder (language model only — the vision tower is intentionally skipped)
    {
        let dir = format!("{root}/text_encoder");
        let cfg: qwen3_image21::qwen3_vl_text::Config =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
        let mut paths: Vec<String> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path().to_string_lossy().to_string()))
            .filter(|p| p.ends_with(".safetensors"))
            .collect();
        paths.sort();
        let (vb, requested, all) = recording_vb(&paths, &device)?;
        let model = qwen3_image21::qwen3_vl_text::Qwen3VLTextEncoder::new(&cfg.text_config, vb)?;
        drop(model);
        report("text encoder (model.language_model.* only)", &all, &requested.lock().unwrap(), |n| {
            n.starts_with("model.language_model.")
        });
    }
    Ok(())
}
