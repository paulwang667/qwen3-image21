//! Diagnostic: verify the Qwen3 stand-in text encoder produces different,
//! shape-correct embeddings for different prompts (i.e. actually depends on
//! the prompt, unlike the random-embedding placeholder it replaces).
use anyhow::Result;
use candle_core::Device;
use qwen3_image21::text_encoder::TextEncoder;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/sn-0523/Desktop/projects/qwen3-jev/models/qwen3-0.6b".to_string());

    let mut encoder = TextEncoder::load(&model_dir, 4096, device)?;

    let a = encoder.encode("a red apple on a table")?;
    let b = encoder.encode("a blue spaceship in outer space")?;
    let c = encoder.encode("a red apple on a table")?;

    println!("a.shape = {:?}", a.shape());
    println!("b.shape = {:?}", b.shape());

    let diff_ab = (&a.mean(1)?.mean(1)? - &b.mean(1)?.mean(1)?)?.abs()?.flatten_all()?.to_vec1::<f32>()?[0];
    let diff_ac = (&a.mean(1)?.mean(1)? - &c.mean(1)?.mean(1)?)?.abs()?.flatten_all()?.to_vec1::<f32>()?[0];
    println!("|mean(a) - mean(b)| (different prompts) = {diff_ab}");
    println!("|mean(a) - mean(c)| (same prompt twice, should be 0)  = {diff_ac}");

    Ok(())
}
