//! Small shared helper for loading possibly-sharded safetensors checkpoints.
use anyhow::Result;

/// Resolves a single safetensors path to every shard of its checkpoint, using
/// the diffusers sharding convention: a file named `<base>-NNNNN-of-MMMMM.safetensors`
/// has a sibling `<base>.safetensors.index.json` whose `weight_map` lists every
/// shard filename. Falls back to just `path` when there's no such index (a
/// single-file checkpoint, which is the common case).
pub fn resolve_safetensors_paths(path: &str) -> Result<Vec<String>> {
    let p = std::path::Path::new(path);
    let dir = p.parent().unwrap_or_else(|| std::path::Path::new("."));
    let stem = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
    if let Some(pos) = stem.find("-00") {
        let base = &stem[..pos];
        let index_path = dir.join(format!("{base}.safetensors.index.json"));
        if index_path.exists() {
            let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let mut shard_names = std::collections::BTreeSet::new();
            if let Some(map) = index.get("weight_map").and_then(|m| m.as_object()) {
                for v in map.values() {
                    if let Some(name) = v.as_str() {
                        shard_names.insert(name.to_string());
                    }
                }
            }
            if !shard_names.is_empty() {
                return Ok(shard_names.into_iter().map(|n| dir.join(n).to_string_lossy().into_owned()).collect());
            }
        }
    }
    Ok(vec![path.to_string()])
}
