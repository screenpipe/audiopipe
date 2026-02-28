# stt

Fast speech-to-text in Rust. ONNX Runtime backend — runs on CoreML (Apple ANE), DirectML (Windows), CUDA, or CPU.

```rust
use stt::{Model, TranscribeOptions};

let model = Model::from_pretrained("parakeet-tdt-0.6b-v2")?;
let result = model.transcribe(&audio_f32, TranscribeOptions::default())?;
println!("{}", result.text);
```

## Models

| Model | Params | Languages | Speed (M4 Max) |
|-------|--------|-----------|----------------|
| `parakeet-tdt-0.6b-v2` | 0.6B | English | ~139x realtime |
| `parakeet-tdt-0.6b-v3` | 0.6B | 25 languages | ~139x realtime |

More coming: Qwen3-ASR, Whisper ONNX.

## Features

```toml
[dependencies]
stt = { version = "0.1", features = ["parakeet"] }
```

Hardware acceleration:

```toml
# macOS — Apple Neural Engine
stt = { version = "0.1", features = ["parakeet", "coreml"] }

# Windows — DirectML (AMD/Intel/NVIDIA)
stt = { version = "0.1", features = ["parakeet", "directml"] }

# NVIDIA — CUDA
stt = { version = "0.1", features = ["parakeet", "cuda"] }
```

## Example

```sh
cargo run --example transcribe --features parakeet -- audio.wav
```

## License

MIT
