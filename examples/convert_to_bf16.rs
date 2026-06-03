//! Offline fp32 -> bf16 safetensors converter.
//!
//! Reads `<input_dir>/model.safetensors`, converts every floating-point tensor
//! to bf16 (F32 and F16 are down-converted, BF16 passes through, any other
//! dtype is copied verbatim), and writes `<output_dir>/model.safetensors`. The
//! companion files (`config.json`, `vocab.txt`, `tokenizer.model`) are copied
//! through when present so the output directory is a complete, loadable model.
//!
//! Distributing the bf16 copy halves the download and lets the native loader
//! (see `parakeet_mlx::load_safetensors`) build weights without ever
//! materializing an fp32 buffer.
//!
//! ```sh
//! cargo run --release --no-default-features --features parakeet-mlx \
//!     --example convert_to_bf16 -- <input_dir> <output_dir> [--verify]
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use half::{bf16, f16};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

/// A converted tensor: name, target dtype, shape, and the owned little-endian
/// byte buffer. The buffers are kept alive in one `Vec` so the borrowed
/// `TensorView`s handed to `serialize_to_file` stay valid until after writing.
struct OwnedTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::BOOL => "BOOL",
        Dtype::U8 => "U8",
        Dtype::I8 => "I8",
        Dtype::F8_E5M2 => "F8_E5M2",
        Dtype::F8_E4M3 => "F8_E4M3",
        Dtype::I16 => "I16",
        Dtype::U16 => "U16",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::I32 => "I32",
        Dtype::U32 => "U32",
        Dtype::F32 => "F32",
        Dtype::F64 => "F64",
        Dtype::I64 => "I64",
        Dtype::U64 => "U64",
        _ => "OTHER",
    }
}

fn print_histogram(label: &str, hist: &BTreeMap<&'static str, usize>) {
    let parts: Vec<String> = hist.iter().map(|(d, n)| format!("{d}={n}")).collect();
    println!("  {label}: {{ {} }}", parts.join(", "));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let positional: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    let verify = args.iter().any(|a| a == "--verify");

    if positional.len() < 2 {
        eprintln!(
            "usage: convert_to_bf16 <input_dir> <output_dir> [--verify]\n\
             reads <input_dir>/model.safetensors, writes <output_dir>/model.safetensors as bf16"
        );
        std::process::exit(2);
    }

    let input_dir = Path::new(positional[0]);
    let output_dir = Path::new(positional[1]);
    std::fs::create_dir_all(output_dir)?;

    let input_st = input_dir.join("model.safetensors");
    let output_st = output_dir.join("model.safetensors");

    let input_size = std::fs::metadata(&input_st)?.len();
    println!(
        "input:  {} ({})",
        input_st.display(),
        human_size(input_size)
    );

    let data = std::fs::read(&input_st)?;
    let tensors = SafeTensors::deserialize(&data)?;

    // Preserve any `__metadata__` block from the source file.
    let (_, metadata) = SafeTensors::read_metadata(&data)?;
    let metadata: Option<std::collections::HashMap<String, String>> = metadata.metadata().clone();

    let mut before: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut after: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut converted: Vec<OwnedTensor> = Vec::new();

    for (name, view) in tensors.tensors() {
        let src_dtype = view.dtype();
        let shape = view.shape().to_vec();
        let bytes = view.data();
        *before.entry(dtype_name(src_dtype)).or_insert(0) += 1;

        let (out_dtype, out_bytes): (Dtype, Vec<u8>) = match src_dtype {
            Dtype::F32 => {
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(4) {
                    let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
                }
                (Dtype::BF16, out)
            }
            Dtype::F16 => {
                let mut out = Vec::with_capacity(bytes.len());
                for chunk in bytes.chunks_exact(2) {
                    let v = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                    out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
                }
                (Dtype::BF16, out)
            }
            Dtype::BF16 => (Dtype::BF16, bytes.to_vec()),
            other => {
                println!(
                    "  passthrough: '{name}' dtype={} (copied verbatim)",
                    dtype_name(other)
                );
                (other, bytes.to_vec())
            }
        };

        *after.entry(dtype_name(out_dtype)).or_insert(0) += 1;
        converted.push(OwnedTensor {
            name,
            dtype: out_dtype,
            shape,
            data: out_bytes,
        });
    }

    let tensor_count = converted.len();

    // Build borrowed views over the owned buffers, then serialize.
    let views: Vec<(&String, TensorView)> = converted
        .iter()
        .map(|t| {
            let view = TensorView::new(t.dtype, t.shape.clone(), &t.data)
                .expect("tensor byte length matches dtype * shape");
            (&t.name, view)
        })
        .collect();

    safetensors::serialize_to_file(views, &metadata, &output_st)?;

    // Copy companion files so the output dir is a complete loadable model.
    for fname in ["config.json", "vocab.txt", "tokenizer.model"] {
        let src = input_dir.join(fname);
        if src.exists() {
            std::fs::copy(&src, output_dir.join(fname))?;
            println!("  copied: {fname}");
        }
    }

    let output_size = std::fs::metadata(&output_st)?.len();
    println!(
        "output: {} ({})",
        output_st.display(),
        human_size(output_size)
    );
    println!("tensors: {tensor_count}");
    print_histogram("before", &before);
    print_histogram("after", &after);
    println!(
        "size: {} -> {} ({:.1}% of original)",
        human_size(input_size),
        human_size(output_size),
        (output_size as f64 / input_size as f64) * 100.0
    );

    if verify {
        verify_load(output_dir)?;
    }

    Ok(())
}

#[cfg(feature = "parakeet-mlx")]
fn verify_load(output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use audiopipe::parakeet_mlx::{mlx_memory_stats, ParakeetMlxEngine};
    use audiopipe::ParakeetPrecision;

    println!(
        "\n--verify: loading bf16 model from {}",
        output_dir.display()
    );
    let (active_before, _) = mlx_memory_stats();
    let _engine =
        ParakeetMlxEngine::from_dir(output_dir, "parakeet-tdt-0.6b-v3", ParakeetPrecision::Bf16)?;
    let (active_after, cache_after) = mlx_memory_stats();
    println!(
        "verify: loaded without panic. mlx active {} -> {} (cache {})",
        human_size(active_before as u64),
        human_size(active_after as u64),
        human_size(cache_after as u64),
    );
    Ok(())
}

#[cfg(not(feature = "parakeet-mlx"))]
fn verify_load(_output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("--verify requires the parakeet-mlx feature");
    Ok(())
}
