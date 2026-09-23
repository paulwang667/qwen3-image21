use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path = Path::new("models/qwen-image-2.1-Q4_0.gguf");
    let device = candle_core::Device::cpu();

    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(gguf_path, &device)?;

    println!("GGUF tensors:");
    println!("---");

    // Try to list some tensor names by accessing them
    let test_names = vec![
        "time_text_embed.timestep_embedder.0.bias",
        "time_text_embed.timestep_embedder.0.weight",
        "double_blocks.0.img_attn.to_q.weight",
        "double_blocks.0.txt_attn.to_q.weight",
        "single_blocks.0.img_attn.to_q.weight",
        "proj_out.weight",
        "norm_out.weight",
        "x_embedder.proj.weight",
    ];

    for name in &test_names {
        match vb.get_no_shape(name) {
            Ok(tensor) => {
                println!("✓ {}", name);
            }
            Err(e) => {
                println!("✗ {} - {}", name, e);
            }
        }
    }

    // Try to find the actual tensor names by checking the gguf file
    let mut file = std::fs::File::open(gguf_path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

    println!("\n\nAll tensor names in GGUF file:");
    println!("---");
    for (name, _) in content.tensor_infos.iter().take(50) {
        println!("{}", name);
    }
    println!("...");
    println!("Total tensors: {}", content.tensor_infos.len());

    Ok(())
}