//! Transcribe the same audio with two Parakeet-MLX models and compare the text:
//!   A: the HF-cache model (fp32 weights, cast to bf16 at runtime)
//!   B: a pre-converted bf16 model directory (loaded natively, no fp32)
//! Both should produce identical transcripts, since the bf16 weights are the
//! same either way. Used to validate `convert_to_bf16` output before shipping.
//!
//! cargo run --release --no-default-features --features parakeet-mlx \
//!     --example parity_check -- <audio.wav> <bf16_model_dir>

use audiopipe::{Model, TranscribeOptions};

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("failed to open wav");
    let spec = reader.spec();
    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    };
    if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples = samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    }
    (samples, spec.sample_rate)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = args.next().expect("usage: parity_check <audio.wav> <bf16_model_dir>");
    let dir = args.next().expect("usage: parity_check <audio.wav> <bf16_model_dir>");
    let (audio, sr) = read_wav(&wav);

    let mut a = Model::from_pretrained_cache_only("parakeet-tdt-0.6b-v3-mlx")
        .expect("load fp32 cache model");
    let ta = a
        .transcribe_with_sample_rate(&audio, sr, TranscribeOptions::default())
        .expect("transcribe A")
        .text;

    let mut b = Model::from_dir(std::path::Path::new(&dir), "parakeet-mlx")
        .expect("load pre-bf16 model dir");
    let tb = b
        .transcribe_with_sample_rate(&audio, sr, TranscribeOptions::default())
        .expect("transcribe B")
        .text;

    println!("A fp32->runtime bf16 : {ta}");
    println!("B pre-bf16 native    : {tb}");
    println!("PARITY: {}", if ta.trim() == tb.trim() { "IDENTICAL" } else { "DIFFER" });
}
