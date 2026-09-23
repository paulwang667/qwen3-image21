//! Custom GGUF VarBuilder with tensor name mapping
//! 
//! Wraps candle_transformers' quantized VarBuilder to apply tensor name mapping
//! from GGUF (original Qwen-Image architecture) to our expected format
//! (diffusers/ComfyUI MMDiT architecture)

use candle_core::quantized::QTensor;
use candle_core::{Device, Result, Shape};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::gguf_mapping::map_gguf_name;

/// Custom GGUF VarBuilder with tensor name mapping
/// 
/// This wraps the standard GGUF VarBuilder and applies tensor name mapping
/// to convert from GGUF's original Qwen-Image naming convention to our expected format.
#[derive(Clone)]
pub struct MappedVarBuilder {
    inner: candle_transformers::quantized_var_builder::VarBuilder,
    /// Reverse mapping: expected_name -> gguf_name
    name_map: HashMap<String, String>,
    device: Device,
}

impl MappedVarBuilder {
    /// Create a new MappedVarBuilder from a GGUF file
    pub fn from_gguf<P: AsRef<Path>>(p: P, device: &Device) -> Result<Self> {
        let path_ref = p.as_ref();
        let inner = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(path_ref, device)?;

        // Build reverse mapping: expected_name -> gguf_name
        let mut name_map = HashMap::new();

        // Read the GGUF file to get all tensor names
        let mut file = std::fs::File::open(path_ref)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        for gguf_name in content.tensor_infos.keys() {
            if let Some(expected_name) = map_gguf_name(gguf_name) {
                name_map.insert(expected_name, gguf_name.clone());
            }
        }

        Ok(Self {
            inner,
            name_map,
            device: device.clone(),
        })
    }

    /// Navigate to a subdirectory (prefix)
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        Self {
            inner: self.inner.pp(s),
            name_map: self.name_map.clone(),
            device: self.device.clone(),
        }
    }

    /// Get a tensor by expected name
    pub fn get<S: Into<Shape> + Clone>(&self, shape: S, name: &str) -> Result<Arc<QTensor>> {
        // Try to find the GGUF name from the expected name
        let gguf_name = if let Some(mapped) = self.name_map.get(name) {
            mapped.clone()
        } else {
            // Try with path prefix from inner VarBuilder
            name.to_string()
        };

        // Try getting with the GGUF name
        if let Ok(tensor) = self.inner.get(shape.clone(), &gguf_name) {
            return Ok(tensor);
        }

        // Fallback: try with the original name
        self.inner.get(shape, name)
    }

    /// Get a tensor by name without shape validation
    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        // Try to find the GGUF name from the expected name
        let gguf_name = if let Some(mapped) = self.name_map.get(name) {
            mapped.clone()
        } else {
            name.to_string()
        };

        // Try getting with the GGUF name
        if let Ok(tensor) = self.inner.get_no_shape(&gguf_name) {
            return Ok(tensor);
        }

        // Fallback: try with the original name
        self.inner.get_no_shape(name)
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if a tensor exists
    pub fn contains_key(&self, key: &str) -> bool {
        // Check with expected name
        if let Some(gguf_name) = self.name_map.get(key) {
            if self.inner.contains_key(gguf_name) {
                return true;
            }
        }
        // Also check with original name
        self.inner.contains_key(key)
    }

    /// Get the underlying inner VarBuilder
    pub fn inner(&self) -> &candle_transformers::quantized_var_builder::VarBuilder {
        &self.inner
    }

    /// Get the name mapping
    pub fn name_map(&self) -> &HashMap<String, String> {
        &self.name_map
    }
}

/// Convenience function to load a GGUF file with name mapping
pub fn load_gguf_mapped<P: AsRef<Path>>(path: P, device: &Device) -> Result<MappedVarBuilder> {
    MappedVarBuilder::from_gguf(path, device)
}

/// Test utility: list all mapped names
pub fn list_mapped_names<P: AsRef<Path>>(path: P) -> Result<Vec<(String, String)>> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    
    let mut mappings = Vec::new();
    for (gguf_name, _) in content.tensor_infos.iter() {
        if let Some(expected_name) = map_gguf_name(gguf_name) {
            mappings.push((gguf_name.clone(), expected_name));
        }
    }
    mappings.sort();
    Ok(mappings)
}