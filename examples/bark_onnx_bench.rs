// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
//! CPU/Windows DirectML benchmark. GPU requests fail if the model fell back to CPU.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Instant};
#[derive(Deserialize)]
struct Case {
    name: String,
    audio: PathBuf,
    reference: String,
}
#[derive(Serialize)]
struct Measurement {
    name: String,
    audio_seconds: f64,
    seconds: Vec<f64>,
    median_seconds: f64,
    transcript: String,
    reference: String,
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() >= 3,
        "bark_onnx_bench bark|parakeet MANIFEST [repeats=3] [cpu|directml[:id]]"
    );
    let repeats: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(3);
    ensure!((1..=100).contains(&repeats), "invalid repeats");
    let requested = a.get(4).map(String::as_str).unwrap_or("cpu");
    let provider = match requested {
        "cpu" => audiopipe::ParakeetExecutionProvider::Cpu,
        "directml" | "directml-or-cpu" => audiopipe::ParakeetExecutionProvider::DirectMl,
        s if s.starts_with("directml:") => {
            audiopipe::ParakeetExecutionProvider::DirectMlDevice(s[9..].parse()?)
        }
        _ => anyhow::bail!("invalid provider"),
    };
    let name = match a[1].as_str() {
        "bark" => "bark-3.0",
        "parakeet" => "parakeet-tdt-0.6b-v3",
        _ => anyhow::bail!("unknown model"),
    };
    let load = Instant::now();
    let mut model = audiopipe::Model::from_pretrained_cache_only_with_provider(name, provider)?;
    let load_seconds = load.elapsed().as_secs_f64();
    eprintln!(
        "requested: {requested}; actual session provider: {:?}",
        model.execution_provider()
    );
    ensure!(
        requested == "cpu"
            || requested == "directml-or-cpu"
            || model.execution_provider() == Some("DirectML"),
        "DirectML request fell back to CPU; refusing to label this a GPU benchmark"
    );
    let cases: Vec<Case> = serde_json::from_reader(std::fs::File::open(&a[2])?)?;
    let mut measurements = Vec::new();
    for c in cases {
        let mut wav = hound::WavReader::open(&c.audio)?;
        let spec = wav.spec();
        ensure!(
            spec.channels == 1 && spec.sample_rate == 16000 && spec.bits_per_sample == 16,
            "expected mono PCM16 16kHz"
        );
        let audio: Vec<f32> = wav
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.))
            .collect::<std::result::Result<_, _>>()?;
        let _warmup = model.transcribe(&audio, Default::default())?;
        let mut seconds = Vec::new();
        let mut transcript = String::new();
        for _ in 0..repeats {
            let start = Instant::now();
            transcript = model.transcribe(&audio, Default::default())?.text;
            seconds.push(start.elapsed().as_secs_f64());
        }
        let mut sorted = seconds.clone();
        sorted.sort_by(f64::total_cmp);
        let median_seconds = sorted[sorted.len() / 2];
        eprintln!("{}: {:.3}s, {}", c.name, median_seconds, transcript);
        measurements.push(Measurement {
            name: c.name,
            audio_seconds: audio.len() as f64 / 16000.,
            seconds,
            median_seconds,
            transcript,
            reference: c.reference,
        });
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"model":name,"backend":"native Rust / ONNX Runtime","requested_provider":requested,"execution_provider":model.execution_provider(),"load_seconds":load_seconds,"cases":measurements})
        )?
    );
    Ok(())
}
