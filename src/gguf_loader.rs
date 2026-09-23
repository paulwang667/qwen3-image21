//! GGUF loader - loads GGUF files and dequantizes to safetensors format
//!
//! This module converts quantized GGUF tensors to standard F32 tensors,
//! saving them as a temporary safetensors file for compatibility with
//! the existing transformer implementation.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Load a GGUF file and dequantize all tensors, saving as safetensors.
/// Returns the path to the temporary safetensors file.
pub fn load_gguf_as_safetensors(
    gguf_path: &Path,
    device: &Device,
    output_path: &Path,
) -> Result<()> {
    eprintln!("Loading GGUF file: {}", gguf_path.display());

    let mut file = std::fs::File::open(gguf_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    eprintln!("Loaded GGUF with {} tensors", content.tensor_infos.len());

    // Prepare to save tensors
    let mut tensors = HashMap::new();

    // Process each tensor
    let mut tensor_count = 0;
    for (tensor_name, _tensor_info) in &content.tensor_infos {
        // Load and dequantize the tensor
        let qtensor = content.tensor(&mut file, tensor_name, device)?;

        // Dequantize to F32
        let tensor = qtensor.dequantize(device)?;

        eprintln!(
            "  Dequantized: {} -> shape={:?}",
            tensor_name,
            tensor.shape()
        );

        // Save to map
        tensors.insert(tensor_name.clone(), tensor);
        tensor_count += 1;

        if tensor_count % 100 == 0 {
            eprintln!("  Processed {} tensors...", tensor_count);
        }
    }

    eprintln!("Total tensors dequantized: {}", tensors.len());
    // Save to safetensors using candle_core's built-in save
    eprintln!("Saving to safetensors: {}", output_path.display());
    candle_core::safetensors::save(&tensors, output_path)?;

    eprintln!("GGUF conversion complete!");

    Ok(())
}

/// Check if a file is a GGUF file
pub fn is_gguf_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "gguf" {
        return true;
    }
    // Check file signature
    if let Ok(mut file) = std::fs::File::open(path) {
        let mut magic = [0u8; 4];
        if file.read(&mut magic).is_ok() {
            return &magic == b"GGUF";
        }
    }
    false
}

/// Convert GGUF tensor names to match our transformer's expected format.
/// The GGUF file may have a "diffusion_model." prefix that needs to be stripped.
pub fn normalize_tensor_name(name: &str) -> String {
    // Strip common prefixes
    let prefixes = ["diffusion_model.", "model.", "transformer."];
    for prefix in &prefixes {
        if name.starts_with(prefix) {
            return name[prefix.len()..].to_string();
        }
    }
    name.to_string()
}