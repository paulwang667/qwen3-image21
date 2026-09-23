//! Test GGUF loading without running full inference
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path = Path::new("models/qwen-image-2.1-Q4_0.gguf");
    let output_path = Path::new("models/qwen-image-2.1-Q4_0.test.safetensors");

    println!("Testing GGUF loading...");
    println!("Input: {}", gguf_path.display());
    println!("Output: {}", output_path.display());

    let device = candle_core::Device::cpu();

    // Test GGUF file detection
    assert!(qwen3_image21::gguf_loader::is_gguf_file(gguf_path), "GGUF detection failed");
    println!("✓ GGUF file detected successfully");

    // Test GGUF to safetensors conversion
    qwen3_image21::gguf_loader::load_gguf_as_safetensors(gguf_path, &device, output_path)?;
    println!("✓ GGUF to safetensors conversion completed");

    // Check output file exists
    assert!(output_path.exists(), "Output safetensors file not created");
    let file_size = std::fs::metadata(output_path)?.len();
    println!("✓ Output file created: {} bytes ({} MB)", file_size, file_size / 1024 / 1024);

    // Clean up test file
    std::fs::remove_file(output_path)?;
    println!("✓ Test file cleaned up");

    println!("\n✓✓✓ GGUF loading test PASSED ✓✓✓");
    Ok(())
}