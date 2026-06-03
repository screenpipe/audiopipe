//! MLX streaming memory-leak probe (dev-only, macOS + `parakeet-mlx`).
//!
//! Replicates how the consumer drives streaming STT: per "dictation" open a
//! `StreamSession`, push growing audio chunks (each push re-decodes the whole
//! buffer), `finish`, drop. Prints the process `phys_footprint` plus MLX's
//! active/cache memory after each dictation.
//!
//! Without cache management the footprint and MLX cache climb every dictation
//! (the bug). With the fix (cache limit + `clear_cache` after finish) they
//! stay flat. Reproduce the OLD behavior in the same build with:
//!   AUDIOPIPE_NO_MLX_CACHE_CLEAR=1 AUDIOPIPE_MLX_CACHE_LIMIT_MB=0
//!
//! Usage:
//!   cargo run --example mlx_stream_leak_probe \
//!     --no-default-features --features parakeet-mlx -- [name] [dictations]

use std::process::Command;
use std::time::Duration;

use audiopipe::parakeet_mlx::mlx_memory_stats;
use audiopipe::{Model, TranscribeOptions};

fn footprint() -> String {
    let pid = std::process::id().to_string();
    Command::new("footprint")
        .arg(&pid)
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .find(|l| l.contains("phys_footprint:"))
                .unwrap_or("phys_footprint: ?")
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "phys_footprint: ?".into())
}

fn report(tag: &str) {
    let (active, cache) = mlx_memory_stats();
    eprintln!(
        "PROBE {tag:<14} {}  mlx_active={}MB mlx_cache={}MB",
        footprint(),
        active >> 20,
        cache >> 20
    );
}

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "parakeet-tdt-0.6b-v3-mlx".to_string());
    let dictations: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);

    eprintln!("PROBE pid={} model={name} dictations={dictations}", std::process::id());
    let mut model = Model::from_pretrained_cache_only(&name).expect("load STT model from HF cache");

    // 0.1s of a 440 Hz tone at 16 kHz, pushed in fine increments (like the
    // real tick cadence). Each push re-decodes the whole, growing buffer, and
    // MLX caches Metal buffers keyed by size — so fine steps + dictations of
    // increasing length keep minting fresh buffer sizes the cache can't reuse.
    let sr = 16_000u32;
    let chunk: Vec<f32> = (0..sr / 10)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / sr as f32).sin() * 0.2)
        .collect();

    report("baseline");
    for d in 0..dictations {
        // Dictation d is (20 + 10*d) chunks long → 2s, 3s, 4s, ... so every
        // dictation re-decodes longer buffers than any before it.
        let pushes = 20 + 10 * d;
        let mut session = model
            .transcribe_stream(sr, TranscribeOptions::default())
            .expect("open stream session");
        for _ in 0..pushes {
            session.push(&chunk).expect("push");
        }
        session.finish().expect("finish");
        drop(session);
        report(&format!("dictation={d} ({}s)", pushes / 10));
    }
    std::thread::sleep(Duration::from_secs(2));
    report("final");
    eprintln!("PROBE done");
}
