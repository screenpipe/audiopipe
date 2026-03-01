# audiopipe

Fast speech-to-text in Rust. Supports ONNX Runtime and GGML backends — runs on Metal (macOS GPU), Vulkan (Windows/Linux GPU), CUDA, CoreML, DirectML, or CPU. No Python, no ffmpeg.

```rust
use audiopipe::{Model, TranscribeOptions};

let mut model = Model::from_pretrained("qwen3-asr-0.6b-ggml")?;
let result = model.transcribe(&audio_f32, 16000, TranscribeOptions::default())?;
println!("{}", result.text);
```

## Models

| Model | Backend | Params | Languages | Notes |
|-------|---------|--------|-----------|-------|
| `qwen3-asr-0.6b-ggml` | GGML | 0.6B | 6000+ | Recommended. Metal/Vulkan/CUDA GPU support |
| `qwen3-asr-0.6b` | ONNX | 0.6B | 6000+ | CoreML/DirectML/CUDA via ONNX Runtime |
| `parakeet-tdt-0.6b-v2` | ONNX | 0.6B | English | NVIDIA Parakeet TDT |
| `parakeet-tdt-0.6b-v3` | ONNX | 0.6B | 25 languages | NVIDIA Parakeet TDT |
| `whisper-*` | whisper.cpp | varies | 99 languages | e.g. `whisper-large-v3-turbo` |

## Usage

### Qwen3-ASR GGML (recommended)

Best balance of speed, accuracy, and language coverage. Uses GGML backend with native GPU support.

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml"] }
```

GPU acceleration:

```toml
# macOS — Metal GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "metal"] }

# Windows/Linux — Vulkan GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "vulkan-ggml"] }

# NVIDIA — CUDA GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "cuda-ggml"] }
```

### Qwen3-ASR ONNX

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr"] }

# With CoreML (macOS)
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr", "coreml"] }

# With DirectML (Windows)
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr", "directml"] }
```

### Parakeet

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["parakeet"] }
```

### Whisper

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["whisper"] }
```

## Feature flags

| Feature | Description |
|---------|-------------|
| `qwen3-asr-ggml` | Qwen3-ASR via GGML (recommended) |
| `qwen3-asr` | Qwen3-ASR via ONNX Runtime |
| `parakeet` | NVIDIA Parakeet TDT via ONNX Runtime |
| `whisper` | Whisper via whisper.cpp |
| `metal` | Metal GPU for GGML models (macOS) |
| `vulkan-ggml` | Vulkan GPU for GGML models (Windows/Linux) |
| `cuda-ggml` | CUDA GPU for GGML models (NVIDIA) |
| `coreml` | CoreML for ONNX models (macOS) |
| `directml` | DirectML for ONNX models (Windows) |
| `cuda` | CUDA for ONNX models (NVIDIA) |

## Examples

```sh
# Transcribe with GGML Qwen3-ASR
cargo run --example test_ggml --features qwen3-asr-ggml -- audio.wav

# Transcribe with Parakeet
cargo run --example transcribe --features parakeet -- audio.wav

# Benchmark
cargo run --example benchmark --features qwen3-asr-ggml --release
```

## Used by

- [screenpipe](https://github.com/screenpipe/screenpipe) — AI screen + audio memory for your desktop

## License

MIT
