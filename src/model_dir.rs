//! Default model locations: the transformer, VAE, and text encoder are looked
//! up in one directory (`--model-dir`) unless their paths are given explicitly.
//!
//! Expected layout (any repo directory name works):
//!
//! ```text
//! models/
//! ├── qwen-image-2.1-Q4_K_M.gguf     # optional GGUF transformer
//! └── Qwen-Image-2.1-official/       # the official repo, or the dir itself
//!     ├── text_encoder/  processor/  vae/
//!     └── transformer/               # only needed without a GGUF
//! ```
use anyhow::{anyhow, bail, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq)]
pub struct ModelPaths {
    /// A `.gguf` file, or the first shard of a safetensors transformer.
    pub transformer: PathBuf,
    pub vae: PathBuf,
    /// `None` when no text encoder was given or found.
    pub text_encoder: Option<PathBuf>,
}

/// Resolves every path not given explicitly from `dir`: the text encoder and
/// VAE from the official repo, the transformer from the only `.gguf` file in
/// `dir`, else from the repo's `transformer/` shards.
pub fn resolve(dir: &Path, transformer: Option<&str>, vae: Option<&str>, text_encoder: Option<&str>) -> Result<ModelPaths> {
    let all_given = transformer.is_some() && vae.is_some() && text_encoder.is_some();
    let repo = if all_given { None } else { official_repo(dir)? };
    let transformer = match transformer {
        Some(p) => PathBuf::from(p),
        None => default_transformer(dir, repo.as_deref())?,
    };
    let vae = match vae {
        Some(p) => PathBuf::from(p),
        None => repo
            .as_ref()
            .map(|r| r.join("vae").join("diffusion_pytorch_model.safetensors"))
            .filter(|p| p.is_file())
            .ok_or_else(|| {
                anyhow!("no VAE found under {}: expected <repo>/vae/diffusion_pytorch_model.safetensors; pass --vae-path or --model-dir", dir.display())
            })?,
    };
    let text_encoder = match text_encoder {
        Some(p) => Some(PathBuf::from(p)),
        None => repo.map(|r| r.join("text_encoder")),
    };
    Ok(ModelPaths { transformer, vae, text_encoder })
}

/// `dir` itself if it holds the official layout, else its only subdirectory
/// that does.
fn official_repo(dir: &Path) -> Result<Option<PathBuf>> {
    let is_repo = |d: &Path| d.join("text_encoder").join("config.json").is_file();
    if is_repo(dir) {
        return Ok(Some(dir.to_path_buf()));
    }
    let mut repos: Vec<PathBuf> = entries(dir).into_iter().filter(|p| p.is_dir() && is_repo(p)).collect();
    match repos.len() {
        0 => Ok(None),
        1 => Ok(repos.pop()),
        _ => bail!(
            "several official repos under {}: {}; pass --model-dir, or --text-encoder-path and --vae-path",
            dir.display(),
            list(&repos)
        ),
    }
}

fn default_transformer(dir: &Path, repo: Option<&Path>) -> Result<PathBuf> {
    let mut ggufs: Vec<PathBuf> = entries(dir).into_iter().filter(|p| has_extension(p, "gguf")).collect();
    match ggufs.len() {
        1 => return Ok(ggufs.remove(0)),
        0 => {}
        _ => bail!("several GGUF transformers in {}: {}; pick one with --model-path", dir.display(), list(&ggufs)),
    }
    // Without a GGUF: the repo's full-precision shards (any one resolves the rest).
    if let Some(shard) = repo.and_then(|r| entries(&r.join("transformer")).into_iter().find(|p| has_extension(p, "safetensors"))) {
        return Ok(shard);
    }
    bail!(
        "no transformer found under {}: expected a .gguf file there or an official repo with transformer/; pass --model-path",
        dir.display()
    )
}

/// Sorted entries of `dir`; empty if it does not exist.
fn entries(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    paths.sort();
    paths
}

fn has_extension(p: &Path, ext: &str) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some(ext)
}

fn list(paths: &[PathBuf]) -> String {
    paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory with the given files (parents created).
    fn tree(name: &str, files: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("qwen_model_dir_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for f in files {
            let p = root.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"").unwrap();
        }
        root
    }

    const REPO: [&str; 3] = [
        "repo/text_encoder/config.json",
        "repo/vae/diffusion_pytorch_model.safetensors",
        "repo/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    ];

    #[test]
    fn prefers_the_gguf_and_finds_the_repo_in_a_subdirectory() {
        let root = tree("gguf", &[&REPO[..], &["q4.gguf"]].concat());
        let paths = resolve(&root, None, None, None).unwrap();
        assert_eq!(paths.transformer, root.join("q4.gguf"));
        assert_eq!(paths.vae, root.join("repo/vae/diffusion_pytorch_model.safetensors"));
        assert_eq!(paths.text_encoder, Some(root.join("repo/text_encoder")));
    }

    #[test]
    fn falls_back_to_the_repo_shards_and_accepts_the_repo_itself() {
        let root = tree("shards", &REPO);
        let paths = resolve(&root.join("repo"), None, None, None).unwrap();
        assert_eq!(paths.transformer, root.join("repo/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"));
        assert_eq!(resolve(&root, None, None, None).unwrap(), paths);
    }

    #[test]
    fn several_ggufs_need_an_explicit_choice() {
        let root = tree("two", &[&REPO[..], &["a.gguf", "b.gguf"]].concat());
        assert!(resolve(&root, None, None, None).unwrap_err().to_string().contains("--model-path"));
        let paths = resolve(&root, Some("b.gguf"), None, None).unwrap();
        assert_eq!(paths.transformer, PathBuf::from("b.gguf"));
    }

    #[test]
    fn explicit_paths_need_no_model_dir() {
        let paths = resolve(Path::new("/nonexistent"), Some("t.gguf"), Some("vae.safetensors"), Some("te")).unwrap();
        assert_eq!(paths.text_encoder, Some(PathBuf::from("te")));
        let err = resolve(Path::new("/nonexistent"), Some("t.gguf"), None, None).unwrap_err();
        assert!(err.to_string().contains("no VAE found"));
    }
}
