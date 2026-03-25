# audiopipe

Fast speech-to-text in Rust. No Python, no FFmpeg.

Supports multiple backends — ONNX Runtime, GGML, MLX, whisper.cpp — with hardware acceleration via Metal (macOS), Vulkan, CUDA, CoreML, DirectML, or CPU.

```rust
use audiopipe::{Model, TranscribeOptions};

let mut model = Model::from_pretrained("parakeet-tdt-0.6b-v3")?;
let result = model.transcribe(&audio_f32, 16000, TranscribeOptions::default())?;
println!("{}", result.text);
```

## Models

| Model | Backend | Params | Languages | Notes |
|-------|---------|--------|-----------|-------|
| `parakeet-tdt-0.6b-v3` | ONNX | 0.6B | 25 | NVIDIA Parakeet TDT. CoreML on macOS, CPU int8 fallback |
| `parakeet-tdt-0.6b-v3-mlx` | MLX | 0.6B | 25 | Parakeet on Apple GPU. 8x faster than Whisper |
| `qwen3-asr-0.6b-antirez` | Pure C | 0.6B | 6000+ | Fastest on CPU. BLAS + AVX2/NEON |
| `qwen3-asr-0.6b-ggml` | GGML | 0.6B | 6000+ | Metal/Vulkan/CUDA GPU support |
| `qwen3-asr-0.6b` | ONNX | 0.6B | 6000+ | CoreML/DirectML/CUDA via ONNX Runtime |
| `whisper-*` | whisper.cpp | varies | 99 | e.g. `whisper-large-v3-turbo` |

All models auto-download from HuggingFace on first use.

## Install

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["parakeet"] }
```

## Usage

### Parakeet MLX (Apple Silicon GPU — fastest)

8x faster than Whisper on Apple Silicon. Uses Metal GPU via Apple's MLX framework.

```toml
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["parakeet-mlx"] }
```

### Parakeet ONNX (cross-platform)

```toml
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["parakeet"] }
```

### Qwen3-ASR antirez (fastest CPU)

Pure C implementation by [antirez](https://github.com/antirez/qwen-asr). Faster-than-realtime on low-end CPUs with BLAS and custom AVX2/NEON kernels.

```toml
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-antirez"] }

# With OpenBLAS (~10x faster matmul on Windows/Linux)
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-antirez-blas"] }
```

macOS uses Accelerate automatically. For Windows OpenBLAS setup:

```powershell
Invoke-WebRequest "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.31/OpenBLAS-0.3.31-x64.zip" -OutFile OpenBLAS.zip
Expand-Archive OpenBLAS.zip -DestinationPath C:\OpenBLAS
[System.Environment]::SetEnvironmentVariable("OPENBLAS_PATH", "C:\OpenBLAS\win64", "User")
```

### Qwen3-ASR GGML (GPU-accelerated)

Best balance of speed, accuracy, and language coverage.

```toml
# CPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml"] }

# macOS Metal GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "metal"] }

# Windows/Linux Vulkan GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "vulkan-ggml"] }

# NVIDIA CUDA GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "cuda-ggml"] }
```

### Qwen3-ASR ONNX

```toml
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr"] }

# With CoreML (macOS) or DirectML (Windows)
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr", "coreml"] }
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr", "directml"] }
```

### Whisper

```toml
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["whisper"] }
```

## Feature flags

### Models

| Feature | Description |
|---------|-------------|
| `parakeet-mlx` | Parakeet TDT via MLX (Apple Silicon GPU) |
| `parakeet` | Parakeet TDT via ONNX Runtime |
| `qwen3-asr-antirez` | Qwen3-ASR pure C (fastest CPU) |
| `qwen3-asr-antirez-blas` | + OpenBLAS for ~10x faster matmul |
| `qwen3-asr-ggml` | Qwen3-ASR via GGML |
| `qwen3-asr` | Qwen3-ASR via ONNX Runtime |
| `whisper` | Whisper via whisper.cpp |

### GPU acceleration

| Feature | Description |
|---------|-------------|
| `metal` | Metal GPU for GGML models (macOS) |
| `vulkan-ggml` | Vulkan GPU for GGML models (Windows/Linux) |
| `cuda-ggml` | CUDA GPU for GGML models (NVIDIA) |
| `coreml` | CoreML for ONNX models (macOS) |
| `directml` | DirectML for ONNX models (Windows) |
| `cuda` | CUDA for ONNX models (NVIDIA) |

## Examples

```sh
# Transcribe with Parakeet
cargo run --release --example transcribe --features parakeet -- audio.wav

# Transcribe with GGML Qwen3-ASR
cargo run --release --example transcribe --features qwen3-asr-ggml -- --model qwen3-asr-0.6b-ggml audio.wav

# Benchmark antirez (pure C, fast CPU)
cargo run --release --example bench_asr --features qwen3-asr-antirez -- audio.wav

# Benchmark with OpenBLAS
cargo run --release --example bench_asr --features qwen3-asr-antirez-blas -- audio.wav
```

## Used by

- [screenpipe](https://github.com/screenpipe/screenpipe) — AI that knows everything you've seen, said, or heard

## License

MIT
