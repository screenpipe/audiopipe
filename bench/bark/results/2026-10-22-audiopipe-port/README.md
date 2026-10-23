# Audiopipe API port verification

Both models ran through `audiopipe::Model::from_pretrained_cache_only` and
`Model::transcribe` in separate native Rust processes, explicitly verified on
`Device(gpu, 0)`. Apple M5 Max, 128 GiB RAM, macOS 26.6.2, Rust 1.88.0.
One excluded warm-up per clip and three measured repetitions per clip.

| Measurement | Bark 3.0 | Parakeet TDT 0.6B v3 |
| --- | ---: | ---: |
| Audio duration, seconds | 170.52 | 170.52 |
| Sum of per-clip median inference times, seconds | 1.274 | 1.227 |
| Diagnostic WER | 18.02% | 15.31% |
| Model load, seconds | 6.67 | 0.43 |

All five final transcripts for each model exactly match their respective earlier
Screenpipe benchmark transcripts. This confirms the port preserves these outputs;
the shared-machine timings are not evidence of an overall model ranking.
[Raw Bark](bark.json), [raw Parakeet](parakeet.json), and
[environment, source hashes and limitations](environment.json) are retained.

This uses the same [upstream MLX fence backport](../2026-10-22-m5-max-metal#toolchain-compatibility-and-reproduction)
required by Xcode 27. No unmodified production Metal build is claimed.
The full library test process crashed with default parallel execution; all
36 library tests, four codec tests and the benchmark test passed when serialized.
The opt-in full-model digest test separately verifies all 699 reconstructed tensors.

```sh
cargo test --no-default-features --features bark-mlx --release --lib --test bark_weights --example bark_parakeet_bench -- --test-threads=1
cargo build --no-default-features --features bark-mlx --release --example bark_parakeet_bench
target/release/examples/bark_parakeet_bench bark bench/bark/fixtures.json 3 gpu
target/release/examples/bark_parakeet_bench parakeet bench/bark/fixtures.json 3 gpu
```
