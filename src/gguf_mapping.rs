//! GGUF tensor name mapping for Qwen-Image-2.1
//!
//! Identity mapping with strict validation: our code expects the exact same tensor names
//! as the Qwen-Image-2.1 checkpoint. The checkpoint uses single-stream architecture
//! with 32 blocks, SwiGLU MLP, and shared modulation.
//!
//! Checkpoint tensor names:
//! - Global: img_in, modulation.1, norm_out.linear, proj_out, time_text_embed.timestep_embedder.linear_1/2,
//!           txt_in.in_layer/out_layer/text_norm
//! - Per-block: transformer_blocks.{i}.attn.to_q/k/v/out.0/norm_q/norm_k, img_mlp.proj/gate_layer/out

/// All tensor names our model expects (identity with checkpoint, validated)
pub fn map_gguf_name(gguf_name: &str) -> Option<String> {
    // Transformer blocks: validate against known submodule patterns
    if let Some(rest) = gguf_name.strip_prefix("transformer_blocks.") {
        let parts: Vec<&str> = rest.splitn(2, '.').collect();
        if parts.len() == 2 {
            if let Ok(_block_idx) = parts[0].parse::<usize>() {
                let submodule = parts[1];
                if is_known_block_submodule(submodule) {
                    return Some(gguf_name.to_string());
                }
            }
        }
    }

    // Global tensors: identity mapping for known names
    match gguf_name {
        "time_text_embed.timestep_embedder.linear_1.weight"
        | "time_text_embed.timestep_embedder.linear_2.weight"
        | "norm_out.linear.weight"
        | "proj_out.weight"
        | "img_in.weight"
        | "txt_in.in_layer.weight"
        | "txt_in.out_layer.weight"
        | "txt_in.text_norm.weight"
        | "modulation.1.weight" => Some(gguf_name.to_string()),
        _ => None,
    }
}

/// Check if a submodule name matches known checkpoint tensors within a transformer block
fn is_known_block_submodule(submodule: &str) -> bool {
    // Attention: to_q, to_k, to_v, to_out.0, norm_q, norm_k
    if submodule.starts_with("attn.to_") {
        let suffix = &submodule[8..]; // skip "attn.to_"
        return matches!(suffix, "q.weight" | "k.weight" | "v.weight" | "out.0.weight");
    }
    if submodule == "attn.norm_q.weight" || submodule == "attn.norm_k.weight" {
        return true;
    }
    // MLP: proj, gate_layer, out
    if matches!(
        submodule,
        "img_mlp.proj.weight" | "img_mlp.gate_layer.weight" | "img_mlp.out.weight"
    ) {
        return true;
    }
    false
}

/// Check if a GGUF name can be mapped
pub fn is_mappable(gguf_name: &str) -> bool {
    map_gguf_name(gguf_name).is_some()
}

/// Get all expected tensor names from GGUF names
pub fn get_expected_names(gguf_names: &[String]) -> Vec<String> {
    gguf_names.iter().filter_map(|name| map_gguf_name(name.as_str())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_global() {
        assert_eq!(
            map_gguf_name("time_text_embed.timestep_embedder.linear_1.weight"),
            Some("time_text_embed.timestep_embedder.linear_1.weight".to_string())
        );
        assert_eq!(
            map_gguf_name("norm_out.linear.weight"),
            Some("norm_out.linear.weight".to_string())
        );
        assert_eq!(
            map_gguf_name("proj_out.weight"),
            Some("proj_out.weight".to_string())
        );
        assert_eq!(
            map_gguf_name("modulation.1.weight"),
            Some("modulation.1.weight".to_string())
        );
    }

    #[test]
    fn test_identity_blocks() {
        assert_eq!(
            map_gguf_name("transformer_blocks.0.attn.to_q.weight"),
            Some("transformer_blocks.0.attn.to_q.weight".to_string())
        );
        assert_eq!(
            map_gguf_name("transformer_blocks.5.attn.norm_k.weight"),
            Some("transformer_blocks.5.attn.norm_k.weight".to_string())
        );
        assert_eq!(
            map_gguf_name("transformer_blocks.31.img_mlp.gate_layer.weight"),
            Some("transformer_blocks.31.img_mlp.gate_layer.weight".to_string())
        );
    }

    #[test]
    fn test_unmappable() {
        assert_eq!(map_gguf_name("unknown.weight"), None);
        assert_eq!(map_gguf_name("transformer_blocks.0.attn.add_q_proj.weight"), None);
        assert_eq!(map_gguf_name("transformer_blocks.0.txt_norm1.weight"), None);
    }
}