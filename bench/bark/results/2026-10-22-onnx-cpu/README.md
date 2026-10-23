# ONNX Runtime CPU validation

Apple M5 Max, 128 GiB RAM. Native Rust, ONNX Runtime CPU, f32 Bark weights,
two intra-op threads and one inter-op thread. Five recordings total 170.52s;
one excluded warm-up per clip and three measured repetitions per clip.

This is a separate backend from the MLX tests. The frontend follows the official
Transformers Parakeet processor and its independent golden fixture. Review the
[raw report](bark.json), [explicit DirectML request recovered to CPU](directml-request-fell-back-to-cpu.json),
and [environment](environment.json). These are not Windows GPU measurements.
The five legacy references have inaccuracies and are not an ASR leaderboard.

Native Windows validation, when available, is recorded in the PR evidence receipt.
Physical DirectML GPU performance is pending available GPU capacity.
