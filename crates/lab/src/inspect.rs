//! `cargo lab inspect <model>`: what's inside a GGUF file, read by the engine's own reader.

use anyhow::Result;
use std::{collections::BTreeMap, path::PathBuf};
use vellum_engine::gguf::Gguf;

pub fn inspect(model: &str, tensors: bool, filter: Option<&str>) -> Result<()> {
    let path = match crate::models::get(model) {
        Ok(model) => model.ensure()?,
        Err(_) => PathBuf::from(model),
    };
    let file = Gguf::open(&path)?;
    println!("{} (GGUF v{})", path.display(), file.version);
    println!("\nmetadata ({} keys)", file.metadata.len());
    for (key, value) in &file.metadata {
        if filter.is_none_or(|f| key.contains(f)) {
            println!("  {key} = {}", value.summary());
        }
    }
    let mut by_type: BTreeMap<String, (usize, u64, usize)> = BTreeMap::new();
    for t in &file.tensors {
        let entry = by_type.entry(format!("{:?}", t.dtype)).or_default();
        entry.0 += 1;
        entry.1 += t.elements();
        entry.2 += t.bytes;
    }
    println!("\ntensors ({})", file.tensors.len());
    for (dtype, (count, elements, bytes)) in &by_type {
        println!(
            "  {dtype:<6} {count:>5} tensors {:>9.1} M parameters {:>8.1} MB",
            *elements as f64 / 1e6,
            *bytes as f64 / 1e6
        );
    }
    if tensors {
        println!();
        for t in &file.tensors {
            if filter.is_none_or(|f| t.name.contains(f)) {
                let dims: Vec<String> = t.dims.iter().map(u64::to_string).collect();
                println!(
                    "  {:<60} {:<6} [{}]",
                    t.name,
                    format!("{:?}", t.dtype),
                    dims.join(", ")
                );
            }
        }
    }
    Ok(())
}
