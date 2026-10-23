# Bark on ONNX Runtime and Windows DirectML

Enable `bark-onnx` and load `bark-3.0` using the standard `Model` API. Add
`directml` on Windows for GPU execution, and `ort-defaults` when using the
standard downloaded ONNX Runtime distribution. Applications supplying their own
ORT distribution may keep their existing runtime configuration.

```rust
Model::download_pretrained("bark-3.0")?;
let mut model = Model::from_pretrained_cache_only_with_provider(
    "bark-3.0", ParakeetExecutionProvider::DirectMl,
)?;
let text = model.transcribe(&audio_f32, TranscribeOptions::default())?.text;
println!("{:?}: {}", model.execution_provider(), text);
```

`DirectMlDevice(id)` selects an exact adapter. `Cpu` requests CPU explicitly.
DirectML uses sequential session execution, disables memory-pattern optimization
and fixes batch/feature/decoder dimensions. Initialization and inference failures
recover to CPU, with the active provider exposed through `execution_provider()`.
Failed CPU recovery is sticky, releases failed sessions and reports unavailable
instead of repeatedly rebuilding sessions or claiming success. DirectML sessions
are recycled after 512 chunks, like the existing Parakeet path.

English only. Contextual keyterm boosting and word timestamps are not implemented.
The `.bark` file is the same pinned 163 MB asset as MLX; no multi-GB Parakeet
weights or Python exporter are downloaded. It expands to f32 in memory.

## Native weight and graph construction

The approximately 782 KB of embedded ONNX architecture graphs contain **no learned
weights**. All 699 native Bark tensors are consumed by the 626 initializer
bindings. Shape/count checks and the pinned `.bark` SHA-256 reject mismatches.
The decoder restores PyTorch IFGO gates to ONNX IOFC and retains both LSTM bias
halves. Convolution/batch-normalization parameters are folded from Bark's own
weights, means, variances and scales. Matrix projections are transposed as needed.
The large sinusoidal position table is generated in Rust, not embedded.

The initializer buffers are passed to ORT in memory and retained for recovery;
there is no converted-weight/model cache. See [template provenance and hashes](../../src/bark/onnx/provenance.json)
and [attribution](../../NOTICE-Bark).

The frontend follows the official Transformers Parakeet processor used by Bark:
constant-centered STFT, symmetric Hann window, Slaney mel filters, log energy,
and sample-variance normalization over real frames. It adds 250 ms of leading
silence. Features are padded after normalization to a fixed maximum shape; only
valid encoder frames are decoded. This differs from the older MLX frontend, so
cross-backend timings/accuracy are separate measurements.

## Reproduce and validate

```sh
cargo test --no-default-features --features bark-onnx,ort-defaults --release --lib --test bark_weights -- --test-threads=1
cargo build --no-default-features --features bark-onnx,ort-defaults --release --example download_bark --example bark_onnx_bench
target/release/examples/download_bark
bash bench/bark/prepare.sh
target/release/examples/bark_onnx_bench bark bench/bark/fixtures.json 3 cpu
```

On a disposable Windows dev VM:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File bench/bark/validate-windows.ps1
```

The Windows validator compiles the DirectML feature natively, verifies all 699
model tensor digests, transcribes five real audio fixtures on CPU and through the
DirectML-or-CPU path, checks expected speech, records native exit codes and GPU
adapter metadata, and performs a strict DirectML request. `directml` benchmark
mode fails if execution fell back to CPU; `directml-or-cpu` explicitly permits
and reports recovery. DirectML registration can use a software adapter, so a
successful session is not automatically labeled a physical GPU benchmark.

The frontend golden fixture was generated independently using Transformers
5.19.0 / PyTorch 2.8.0. Rust tests require no Python. Original ONNX architecture
graphs are pinned to `istupakov/parakeet-tdt-0.6b-v3-onnx` revision
`8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`. To regenerate the stripped templates,
download only `encoder-model.onnx` and `decoder_joint-model.onnx` at that revision
(the external encoder weight data is not needed), then run this **Rust** tool:

```sh
cargo run --no-default-features --features bark --example prepare_bark_onnx -- encoder-model.onnx decoder_joint-model.onnx work/regenerated-templates
```

Hardware DirectML performance remains unverified until a physical Windows GPU is
available. The available Azure validation subscription currently has no GPU quota
in the validated image's replicated regions. CPU/Metal results elsewhere in this
repository must not be presented as Windows GPU measurements.
