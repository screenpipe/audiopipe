# Native Bark 3.0

Enable `bark-mlx` to use `Model::from_pretrained("bark-3.0-mlx")` on Apple silicon.
The native Rust ANS/XZ reader reconstructs the pinned compressed weights, verifies
SHA-256 and the full converted MLX tensor layout, then runs the shared Parakeet
Conformer, prediction and joint networks with Bark's greedy TDT decoder.
No Python, converted-weight cache, transcription server or subprocess is used.

`Model::from_pretrained_cache_only` performs no network requests and returns
`Error::ModelNotCached` while files are absent. `Model::spawn_pretrained_download`
uses audiopipe's existing single-flight download coordinator and creates no
inference state. Only the chosen engine is loaded. Bark model construction and
both Bark/Parakeet inference use the same process-wide GPU lock. Bark drains the
selected MLX stream before returning, including errors.

Bark is English only. Input is resampled to 16 kHz and split into 30-second fixed
shapes, with upstream's 250 ms leading silence. Empty/digital-silence input returns
empty text. Word timestamps and contextual keyterm boosting are not implemented;
results have empty segments. Weights expand to f32: a 163 MB download does not
mean a 163 MB runtime footprint.

Pinned source: `ConwayResearch/Underdog-Bark-3.0`, revision
`85db2ad18ab41e534a222ee253c29191d77ed919`. Downloaded files are `model.bark`,
`mlx/config.json` and `NOTICE`. Attribution is retained in [NOTICE-Bark](../../NOTICE-Bark).

## Validation and benchmark

```sh
cargo test --no-default-features --features bark --test bark_weights
BARK_MODEL_PATH=/path/to/model.bark cargo test --no-default-features --features bark --release --test bark_weights -- --ignored
cargo test --no-default-features --features bark-mlx --release --lib --test bark_weights --example bark_parakeet_bench -- --test-threads=1
bash bench/bark/prepare.sh
cargo build --no-default-features --features bark-mlx --release --example bark_parakeet_bench
target/release/examples/bark_parakeet_bench bark bench/bark/fixtures.json 10 gpu
target/release/examples/bark_parakeet_bench parakeet bench/bark/fixtures.json 10 gpu
```

Both model caches must already be populated for the benchmark. To download a
missing model through the API, call `Model::download_pretrained` with
`bark-3.0-mlx` or `parakeet-tdt-0.6b-v3-mlx`. Compare separate processes with the
same audio and feature set. GPU mode explicitly selects and verifies `Device(gpu, 0)`.
CPU mode is available with the final `cpu` argument. Timing covers model inference,
not capture, VAD or indexing. Case/punctuation normalization is used for diagnostic
WER; the legacy references contain inaccuracies and do not establish overall
model accuracy.

The [archived CPU/GPU reports](results/2026-10-22-m5-max-metal-repeat) were measured
before moving the implementation from Screenpipe. Their manifests retain the
original repository, source revisions and hashes; they are not a performance
measurement of this port. The current port's [API and GPU verification](results/2026-10-22-audiopipe-port)
is recorded separately.

MLX 0.25.1 requires [upstream fence fix 47430159](https://github.com/ml-explore/mlx/commit/47430159fc5f82392bcc69d41fce2ee89fc2b662)
when compiling with Xcode 27's Metal Toolchain. After its `fence.metal` compilation
failure, apply the exact patch to this checkout's generated dependency, then retry:

```sh
patch -d <target/release/build/mlx-sys-*/out/build/_deps/mlx-src> -p1 < bench/bark/results/2026-10-22-m5-max-metal/mlx-fence.patch
```

The benchmark uses that documented compatibility backport for both models. It
does not validate an unmodified production dependency build on Xcode 27. No MLX
upgrade or global source change is included here.
