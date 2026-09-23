use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path = Path::new("models/qwen-image-2.1-Q4_0.gguf");
    let device = candle_core::Device::cuda_if_available(0)?;

    let mut file = std::fs::File::open(gguf_path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

    println!("Total tensors: {}", content.tensor_infos.len());
    println!("---");
    for (name, info) in content.tensor_infos.iter() {
        let dims: Vec<usize> = info.shape.dims().iter().copied().collect();
        println!("{} -> dims={:?}", name, dims);
    }

    Ok(())
}