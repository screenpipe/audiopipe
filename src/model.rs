// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com

use crate::error::{Error, Result};
use std::collections::HashSet;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Global GPU serialization lock for Metal/MLX operations.
///
/// MLX uses asynchronous Metal command buffers. Even after `transcribe()` returns,
/// GPU work may still be in-flight on dispatch queues. If another transcription
/// submits commands before the previous ones complete, the Metal command buffer
/// handler can fail and call `abort()` — killing the entire process.
///
/// This lock ensures only ONE GPU transcription runs at any time across the entire
/// process, regardless of which Model instance or thread initiates it.
/// ONNX (CoreML/DirectML/CPU) and cloud API engines are NOT affected.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Result of asking audiopipe to start a background model download.
///
/// Calls for the same model are coalesced process-wide while a download is in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretrainedDownloadStatus {
    /// This call started the background worker.
    Started,
    /// A worker for this model is already running.
    AlreadyInProgress,
    /// The operating system refused to create the background thread.
    SpawnFailed,
}

#[derive(Default)]
struct DownloadState {
    in_flight: Mutex<HashSet<String>>,
    changed: Condvar,
}

#[derive(Clone, Default)]
struct DownloadCoordinator {
    state: Arc<DownloadState>,
}

struct InFlightDownload {
    key: String,
    state: Arc<DownloadState>,
}

impl Drop for InFlightDownload {
    fn drop(&mut self) {
        let mut in_flight = self
            .state
            .in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        in_flight.remove(&self.key);
        drop(in_flight);
        self.state.changed.notify_all();
    }
}

impl DownloadCoordinator {
    fn spawn_model<F>(&self, name: &str, download: F) -> PretrainedDownloadStatus
    where
        F: FnOnce() -> Result<()> + Send + 'static,
    {
        self.spawn(pretrained_download_key(name).to_string(), download)
    }

    fn spawn<F>(&self, key: String, download: F) -> PretrainedDownloadStatus
    where
        F: FnOnce() -> Result<()> + Send + 'static,
    {
        {
            let mut in_flight = self
                .state
                .in_flight
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !in_flight.insert(key.clone()) {
                return PretrainedDownloadStatus::AlreadyInProgress;
            }
        }

        let state = Arc::clone(&self.state);
        let worker_key = key.clone();
        let spawn_result = std::thread::Builder::new()
            .name("audiopipe-hf-download".to_string())
            .spawn(move || {
                // Remove the in-flight marker on success, error, or panic so a
                // later call can always retry.
                let _in_flight = InFlightDownload {
                    key: worker_key.clone(),
                    state,
                };
                match download() {
                    Ok(()) => tracing::info!(
                        "audiopipe: pretrained model artifacts ready: {}",
                        worker_key
                    ),
                    Err(error) => tracing::warn!(
                        "audiopipe: pretrained model download failed for {}: {}",
                        worker_key,
                        error
                    ),
                }
            });

        if let Err(error) = spawn_result {
            let mut in_flight = self
                .state
                .in_flight
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            in_flight.remove(&key);
            drop(in_flight);
            self.state.changed.notify_all();
            tracing::warn!(
                "audiopipe: failed to spawn pretrained download thread for {}: {}",
                key,
                error
            );
            PretrainedDownloadStatus::SpawnFailed
        } else {
            PretrainedDownloadStatus::Started
        }
    }

    #[cfg(test)]
    fn wait_until_idle(&self, key: &str, timeout: std::time::Duration) -> bool {
        let in_flight = self
            .state
            .in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (in_flight, _) = self
            .state
            .changed
            .wait_timeout_while(in_flight, timeout, |in_flight| in_flight.contains(key))
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !in_flight.contains(key)
    }
}

fn pretrained_downloads() -> &'static DownloadCoordinator {
    static DOWNLOADS: OnceLock<DownloadCoordinator> = OnceLock::new();
    DOWNLOADS.get_or_init(DownloadCoordinator::default)
}

/// Canonical identity for one set of downloadable artifacts.
///
/// Some public model names are aliases for exactly the same Hugging Face files. They must share
/// an in-flight key or concurrent callers can still create duplicate download workers.
fn pretrained_download_key(name: &str) -> &str {
    match name {
        "qwen3-asr-antirez" => "qwen3-asr-0.6b-antirez",
        "qwen3-asr-0.6b-ggml-f16" => "qwen3-asr-0.6b-ggml",
        _ => name,
    }
}

/// A loaded STT model ready for inference.
pub struct Model {
    inner: Box<dyn Engine + Send>,
    /// Whether this model uses a GPU backend that needs serialization.
    uses_gpu: bool,
}

/// Requested execution provider for ONNX Parakeet models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParakeetExecutionProvider {
    /// Use the CPU execution provider.
    #[default]
    Cpu,
    /// Try DirectML's high-performance GPU selection, then fall back to CPU.
    DirectMl,
    /// Use the exact DirectML adapter ordinal selected by the caller.
    DirectMlDevice(i32),
}

/// Requested execution provider for ONNX Qwen3-ASR models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Qwen3ExecutionProvider {
    /// Automatically use DirectML on Windows when it is compiled in.
    #[default]
    Auto,
    /// Use the CPU execution provider.
    Cpu,
    /// Use the exact DirectML adapter ordinal selected by the caller.
    DirectMlDevice(i32),
}

impl ParakeetExecutionProvider {
    pub(crate) fn legacy_default() -> Self {
        #[cfg(all(target_os = "windows", feature = "directml"))]
        if std::env::var("SCREENPIPE_DIRECTML")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        {
            return Self::DirectMl;
        }
        Self::Cpu
    }
}

/// Engine trait — implemented by each backend (Parakeet, Whisper, etc.).
pub(crate) trait Engine: Send + Sync {
    fn transcribe(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult>;
    fn name(&self) -> &str;
    fn execution_provider(&self) -> Option<&'static str> {
        None
    }
    fn fallback_to_cpu(&mut self) -> Result<bool> {
        Ok(false)
    }
}

// Minimal fake backend for exercising public download routing without network I/O.
#[cfg(test)]
const TEST_PRETRAINED_MODEL: &str = "audiopipe-test-pretrained";
#[cfg(test)]
static TEST_DOWNLOADS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static TEST_ENGINE_CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static TEST_DOWNLOAD_RELEASED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
struct TestPretrainedEngine;

#[cfg(test)]
impl TestPretrainedEngine {
    fn from_pretrained() -> Self {
        TEST_ENGINE_CONSTRUCTIONS.fetch_add(1, Ordering::SeqCst);
        Self
    }

    fn download_pretrained() -> Result<()> {
        TEST_DOWNLOADS.fetch_add(1, Ordering::SeqCst);
        while !TEST_DOWNLOAD_RELEASED.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        Ok(())
    }
}

#[cfg(test)]
impl Engine for TestPretrainedEngine {
    fn transcribe(
        &mut self,
        _audio: &[f32],
        _sample_rate: u32,
        _opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        unreachable!("the test download backend must never be used for inference")
    }

    fn name(&self) -> &str {
        TEST_PRETRAINED_MODEL
    }
}

/// Options for transcription.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Language code (e.g. "en"). None = auto-detect.
    pub language: Option<String>,
    /// Return word-level timestamps.
    pub word_timestamps: bool,
    /// Phrases to bias the decoder toward — contextual biasing / "keyterms"
    /// (attendee names, product/company terms, jargon). Honored by the Parakeet
    /// TDT engine via shallow-fusion logit boosting in greedy decode; other
    /// engines currently ignore it. No retraining required.
    pub keyterms: Vec<String>,
    /// Additive logit boost applied to each token that advances a keyterm match
    /// (shallow-fusion weight). Ignored when `keyterms` is empty or <= 0.
    ///
    /// Measured on the edge-case WER corpus: ~6 is optimal (overall WER −2.2pt,
    /// proper-name WER 21%→7%, product 42%→25%). The safe range is ~4-8; at >=10
    /// the decoder over-inserts keyterm tokens and WER degrades sharply, so
    /// callers should keep this modest.
    pub keyterm_boost: f32,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            word_timestamps: false,
            keyterms: Vec::new(),
            keyterm_boost: 6.0,
        }
    }
}

/// Transcription result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscribeResult {
    /// Full transcription text.
    pub text: String,
    /// Per-segment breakdown (if available).
    pub segments: Vec<Segment>,
}

/// A timed segment of transcription.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Segment {
    pub start_secs: f64,
    pub end_secs: f64,
    pub text: String,
}

impl Model {
    /// Load a model by name. Downloads from HuggingFace if not cached.
    ///
    /// Supported models:
    /// - `"parakeet-tdt-0.6b-v2"` — NVIDIA Parakeet TDT (English, 0.6B)
    /// - `"parakeet-tdt-0.6b-v3"` — NVIDIA Parakeet TDT (25 languages, 0.6B)
    /// - `"whisper-large-v3-turbo"` — OpenAI Whisper large-v3-turbo
    /// - `"whisper-tiny"` / `"whisper-small"` / etc.
    pub fn from_pretrained(name: &str) -> Result<Self> {
        match name {
            #[cfg(test)]
            TEST_PRETRAINED_MODEL => Ok(Self {
                inner: Box::new(TestPretrainedEngine::from_pretrained()),
                uses_gpu: false,
            }),
            #[cfg(feature = "parakeet-mlx")]
            n if n.contains("mlx") && n.starts_with("parakeet") => {
                let base_name = n.replace("-mlx", "");
                let engine = crate::parakeet_mlx::ParakeetMlxEngine::from_pretrained(&base_name)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: true })
            }
            #[cfg(feature = "parakeet")]
            n if n.starts_with("parakeet") => {
                let engine = crate::parakeet::ParakeetEngine::from_pretrained_with_provider(
                    n,
                    ParakeetExecutionProvider::legacy_default(),
                )?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "whisper")]
            n if n.starts_with("whisper") => {
                let engine = crate::whisper::WhisperEngine::from_pretrained(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "qwen3-asr-antirez")]
            n if n.starts_with("qwen3-asr") && n.contains("antirez") => {
                let engine = crate::qwen3_asr_antirez::AntirezAsrEngine::from_pretrained(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "qwen3-asr-ggml")]
            n if n.starts_with("qwen3-asr") && n.contains("ggml") => {
                let engine = crate::qwen3_asr_ggml::Qwen3AsrGgmlEngine::from_pretrained(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "qwen3-asr")]
            n if n.starts_with("qwen3-asr") => {
                let engine = crate::qwen3_asr::Qwen3AsrEngine::from_pretrained(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            _ => Err(Error::ModelNotFound(format!(
                "unknown model '{}'. available: parakeet-tdt-0.6b-v2, parakeet-tdt-0.6b-v3, qwen3-asr-0.6b, whisper-*",
                name
            ))),
        }
    }

    /// Load only if every weight file is already in the Hugging Face hub cache (no network I/O).
    /// Returns [`Error::ModelNotCached`] if a download is still needed — use [`Self::spawn_pretrained_download`]
    /// then retry later with this method or [`Self::from_pretrained`].
    pub fn from_pretrained_cache_only(name: &str) -> Result<Self> {
        Self::from_pretrained_cache_only_with_provider(
            name,
            ParakeetExecutionProvider::legacy_default(),
        )
    }

    /// Load a cached model with an explicit Parakeet execution-provider preference.
    /// Non-Parakeet models ignore `provider`.
    pub fn from_pretrained_cache_only_with_provider(
        name: &str,
        provider: ParakeetExecutionProvider,
    ) -> Result<Self> {
        match name {
            #[cfg(feature = "parakeet-mlx")]
            n if n.contains("mlx") && n.starts_with("parakeet") => {
                let base_name = n.replace("-mlx", "");
                let engine =
                    crate::parakeet_mlx::ParakeetMlxEngine::from_pretrained_cache_only(&base_name)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: true })
            }
            #[cfg(feature = "parakeet")]
            n if n.starts_with("parakeet") => {
                let engine = crate::parakeet::ParakeetEngine::from_pretrained_cache_only_with_provider(n, provider)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "whisper")]
            n if n.starts_with("whisper") => {
                let engine = crate::whisper::WhisperEngine::from_pretrained_cache_only(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "qwen3-asr-antirez")]
            n if n.starts_with("qwen3-asr") && n.contains("antirez") => {
                let engine = crate::qwen3_asr_antirez::AntirezAsrEngine::from_pretrained_cache_only(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "qwen3-asr-ggml")]
            n if n.starts_with("qwen3-asr") && n.contains("ggml") => {
                let engine = crate::qwen3_asr_ggml::Qwen3AsrGgmlEngine::from_pretrained_cache_only(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            #[cfg(feature = "qwen3-asr")]
            n if n.starts_with("qwen3-asr") => {
                let engine = crate::qwen3_asr::Qwen3AsrEngine::from_pretrained_cache_only(n)?;
                Ok(Self { inner: Box::new(engine), uses_gpu: false })
            }
            _ => Err(Error::ModelNotFound(format!(
                "unknown model '{}'. available: parakeet-tdt-0.6b-v2, parakeet-tdt-0.6b-v3, qwen3-asr-0.6b, whisper-*",
                name
            ))),
        }
    }

    /// Load cached Qwen3-ASR with an explicit execution provider.
    pub fn from_pretrained_cache_only_with_qwen3_provider(
        name: &str,
        provider: Qwen3ExecutionProvider,
    ) -> Result<Self> {
        #[cfg(feature = "qwen3-asr")]
        if name.starts_with("qwen3-asr") && !name.contains("antirez") && !name.contains("ggml") {
            let engine =
                crate::qwen3_asr::Qwen3AsrEngine::from_pretrained_cache_only_with_provider(
                    name, provider,
                )?;
            return Ok(Self {
                inner: Box::new(engine),
                uses_gpu: false,
            });
        }
        let _ = provider;
        Err(Error::ModelNotFound(format!(
            "unknown ONNX Qwen3-ASR model '{name}'"
        )))
    }

    /// Download all artifacts required by a pretrained model without constructing its engine.
    ///
    /// This is blocking. Use [`Self::spawn_pretrained_download`] when the caller must not wait
    /// for network I/O. Unlike [`Self::from_pretrained`], this method never creates MLX, ONNX,
    /// Whisper, or GGML inference state.
    pub fn download_pretrained(name: &str) -> Result<()> {
        match name {
            #[cfg(test)]
            TEST_PRETRAINED_MODEL => TestPretrainedEngine::download_pretrained(),
            #[cfg(feature = "parakeet-mlx")]
            n if n.contains("mlx") && n.starts_with("parakeet") => {
                let base_name = n.replace("-mlx", "");
                crate::parakeet_mlx::ParakeetMlxEngine::download_pretrained(&base_name)
            }
            #[cfg(feature = "parakeet")]
            n if n.starts_with("parakeet") => {
                crate::parakeet::ParakeetEngine::download_pretrained(n)
            }
            #[cfg(feature = "whisper")]
            n if n.starts_with("whisper") => {
                crate::whisper::WhisperEngine::download_pretrained(n)
            }
            #[cfg(feature = "qwen3-asr-antirez")]
            n if n.starts_with("qwen3-asr") && n.contains("antirez") => {
                crate::qwen3_asr_antirez::AntirezAsrEngine::download_pretrained(n)
            }
            #[cfg(feature = "qwen3-asr-ggml")]
            n if n.starts_with("qwen3-asr") && n.contains("ggml") => {
                crate::qwen3_asr_ggml::Qwen3AsrGgmlEngine::download_pretrained(n)
            }
            #[cfg(feature = "qwen3-asr")]
            n if n.starts_with("qwen3-asr") => {
                crate::qwen3_asr::Qwen3AsrEngine::download_pretrained(n)
            }
            _ => Err(Error::ModelNotFound(format!(
                "unknown model '{}'. available: parakeet-tdt-0.6b-v2, parakeet-tdt-0.6b-v3, qwen3-asr-0.6b, whisper-*",
                name
            ))),
        }
    }

    /// Start a process-wide, single-flight background artifact download for a model.
    ///
    /// Repeated calls for the same model while its worker is running return
    /// [`PretrainedDownloadStatus::AlreadyInProgress`]. Aliases that resolve to the same artifact
    /// set share that in-flight worker. The entry is removed after success, error, or panic so
    /// future calls can retry. This only populates the local cache; it does not construct an
    /// inference engine or allocate model weights.
    pub fn spawn_pretrained_download(name: impl Into<String>) -> PretrainedDownloadStatus {
        let name = name.into();
        let download_name = name.clone();
        pretrained_downloads().spawn_model(&name, move || Self::download_pretrained(&download_name))
    }

    /// Load a model from a local directory containing ONNX files.
    pub fn from_dir(path: &std::path::Path, engine_type: &str) -> Result<Self> {
        match engine_type {
            #[cfg(feature = "parakeet")]
            "parakeet" => {
                let engine = crate::parakeet::ParakeetEngine::from_dir(path)?;
                Ok(Self {
                    inner: Box::new(engine),
                    uses_gpu: false,
                })
            }
            #[cfg(feature = "whisper")]
            "whisper" => {
                let engine = crate::whisper::WhisperEngine::from_dir(path)?;
                Ok(Self {
                    inner: Box::new(engine),
                    uses_gpu: false,
                })
            }
            #[cfg(feature = "qwen3-asr")]
            "qwen3-asr" => {
                let engine = crate::qwen3_asr::Qwen3AsrEngine::from_dir(path)?;
                Ok(Self {
                    inner: Box::new(engine),
                    uses_gpu: false,
                })
            }
            #[cfg(feature = "qwen3-asr-ggml")]
            "qwen3-asr-ggml" => {
                let engine = crate::qwen3_asr_ggml::Qwen3AsrGgmlEngine::from_dir(path)?;
                Ok(Self {
                    inner: Box::new(engine),
                    uses_gpu: false,
                })
            }
            #[cfg(feature = "qwen3-asr-antirez")]
            "qwen3-asr-antirez" => {
                let engine = crate::qwen3_asr_antirez::AntirezAsrEngine::from_dir(path)?;
                Ok(Self {
                    inner: Box::new(engine),
                    uses_gpu: false,
                })
            }
            _ => Err(Error::ModelNotFound(format!(
                "unknown engine type '{}'",
                engine_type
            ))),
        }
    }

    /// Transcribe 16kHz f32 mono audio.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        opts: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let _guard = self.acquire_gpu_lock();
        let first = self.inner.transcribe(audio, 16000, &opts);
        self.retry_after_provider_failure(first, audio, 16000, &opts)
    }

    /// Transcribe audio at a given sample rate (resampled internally to 16kHz).
    pub fn transcribe_with_sample_rate(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        opts: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let _guard = self.acquire_gpu_lock();
        let resampled;
        let audio = if sample_rate == 16000 {
            audio
        } else {
            resampled = crate::audio::resample(audio, sample_rate, 16000);
            &resampled
        };
        let first = self.inner.transcribe(audio, 16000, &opts);
        self.retry_after_provider_failure(first, audio, 16000, &opts)
    }

    fn retry_after_provider_failure(
        &mut self,
        first: Result<TranscribeResult>,
        audio: &[f32],
        sample_rate: u32,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let gpu_error = match first {
            Ok(result) => return Ok(result),
            Err(error) => error,
        };
        match self.inner.fallback_to_cpu() {
            Ok(true) => {
                tracing::warn!(
                    "audiopipe: GPU inference failed ({gpu_error}); CPU recovery initialized; retrying the same audio"
                );
                match self.inner.transcribe(audio, sample_rate, opts) {
                    Ok(result) => {
                        tracing::warn!(
                            "audiopipe: CPU retry completed after GPU inference failure ({gpu_error})"
                        );
                        Ok(result)
                    }
                    Err(cpu_error) => {
                        tracing::error!(
                            "audiopipe: CPU retry failed ({cpu_error}) after GPU inference failure ({gpu_error})"
                        );
                        Err(Error::Other(format!(
                            "GPU inference failed ({gpu_error}); CPU retry failed ({cpu_error})"
                        )))
                    }
                }
            }
            Ok(false) => Err(gpu_error),
            Err(cpu_init_error) => {
                tracing::error!(
                    "audiopipe: CPU recovery initialization failed ({cpu_init_error}) after GPU inference failure ({gpu_error})"
                );
                Err(Error::Other(format!(
                    "GPU inference failed ({gpu_error}); CPU recovery initialization failed ({cpu_init_error})"
                )))
            }
        }
    }

    /// Acquire the GPU lock if this model uses a GPU backend.
    /// Returns None for CPU/ONNX backends (no serialization needed).
    fn acquire_gpu_lock(&self) -> Option<std::sync::MutexGuard<'static, ()>> {
        if !self.uses_gpu {
            return None;
        }
        match GPU_LOCK.lock() {
            Ok(guard) => Some(guard),
            Err(poisoned) => Some(poisoned.into_inner()),
        }
    }

    /// Model name.
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Active execution provider, when the backend exposes one.
    pub fn execution_provider(&self) -> Option<&'static str> {
        self.inner.execution_provider()
    }
}

#[cfg(test)]
mod download_tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_secs(5);
    type ReleaseGate = Arc<(Mutex<bool>, Condvar)>;

    fn wait_for_release(gate: &ReleaseGate) {
        let (released, changed) = &**gate;
        let mut released = released
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while !*released {
            released = changed
                .wait(released)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn release(gate: &ReleaseGate) {
        let (released, changed) = &**gate;
        *released
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        changed.notify_all();
    }

    #[test]
    fn public_background_download_is_single_flight_and_download_only() {
        const CALLERS: usize = 16;

        TEST_DOWNLOADS.store(0, Ordering::SeqCst);
        TEST_ENGINE_CONSTRUCTIONS.store(0, Ordering::SeqCst);
        TEST_DOWNLOAD_RELEASED.store(false, Ordering::SeqCst);
        let caller_barrier = Arc::new(Barrier::new(CALLERS));

        let callers: Vec<_> = (0..CALLERS)
            .map(|_| {
                let caller_barrier = Arc::clone(&caller_barrier);
                std::thread::spawn(move || {
                    caller_barrier.wait();
                    Model::spawn_pretrained_download(TEST_PRETRAINED_MODEL)
                })
            })
            .collect();

        let statuses: Vec<_> = callers
            .into_iter()
            .map(|caller| caller.join().unwrap())
            .collect();

        let deadline = std::time::Instant::now() + TIMEOUT;
        while TEST_DOWNLOADS.load(Ordering::SeqCst) == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        let downloads = TEST_DOWNLOADS.load(Ordering::SeqCst);
        let engine_constructions = TEST_ENGINE_CONSTRUCTIONS.load(Ordering::SeqCst);

        // Always let the detached worker exit before asserting so a failed
        // assertion cannot leave the process-global test worker blocked.
        TEST_DOWNLOAD_RELEASED.store(true, Ordering::SeqCst);
        let worker_exited = pretrained_downloads().wait_until_idle(TEST_PRETRAINED_MODEL, TIMEOUT);

        assert!(worker_exited);
        assert_eq!(downloads, 1);
        assert_eq!(engine_constructions, 0);
        assert_eq!(
            statuses
                .iter()
                .filter(|status| **status == PretrainedDownloadStatus::Started)
                .count(),
            1
        );
        assert_eq!(
            statuses
                .iter()
                .filter(|status| **status == PretrainedDownloadStatus::AlreadyInProgress)
                .count(),
            CALLERS - 1
        );
    }

    #[test]
    fn concurrent_calls_for_one_model_start_one_worker() {
        const CALLERS: usize = 16;

        let coordinator = Arc::new(DownloadCoordinator::default());
        let caller_barrier = Arc::new(Barrier::new(CALLERS));
        let worker_count = Arc::new(AtomicUsize::new(0));
        let release_gate: ReleaseGate = Arc::new((Mutex::new(false), Condvar::new()));
        let (worker_started, worker_entered) = mpsc::sync_channel(1);

        let callers: Vec<_> = (0..CALLERS)
            .map(|_| {
                let coordinator = Arc::clone(&coordinator);
                let caller_barrier = Arc::clone(&caller_barrier);
                let worker_count = Arc::clone(&worker_count);
                let release_gate = Arc::clone(&release_gate);
                let worker_started = worker_started.clone();
                std::thread::spawn(move || {
                    caller_barrier.wait();
                    coordinator.spawn("same-model".to_string(), move || {
                        worker_count.fetch_add(1, Ordering::SeqCst);
                        worker_started.send(()).unwrap();
                        wait_for_release(&release_gate);
                        Ok(())
                    })
                })
            })
            .collect();

        let statuses: Vec<_> = callers
            .into_iter()
            .map(|caller| caller.join().unwrap())
            .collect();
        worker_entered.recv_timeout(TIMEOUT).unwrap();

        assert_eq!(
            statuses
                .iter()
                .filter(|status| **status == PretrainedDownloadStatus::Started)
                .count(),
            1
        );
        assert_eq!(
            statuses
                .iter()
                .filter(|status| **status == PretrainedDownloadStatus::AlreadyInProgress)
                .count(),
            CALLERS - 1
        );
        assert_eq!(worker_count.load(Ordering::SeqCst), 1);

        release(&release_gate);
        assert!(coordinator.wait_until_idle("same-model", TIMEOUT));
    }

    #[test]
    fn different_models_run_independently() {
        let coordinator = DownloadCoordinator::default();
        let release_gate: ReleaseGate = Arc::new((Mutex::new(false), Condvar::new()));
        let (worker_started, worker_entered) = mpsc::sync_channel(2);

        let first_gate = Arc::clone(&release_gate);
        let first_started = worker_started.clone();
        assert_eq!(
            coordinator.spawn("model-a".to_string(), move || {
                first_started.send("model-a").unwrap();
                wait_for_release(&first_gate);
                Ok(())
            }),
            PretrainedDownloadStatus::Started
        );

        let second_gate = Arc::clone(&release_gate);
        assert_eq!(
            coordinator.spawn("model-b".to_string(), move || {
                worker_started.send("model-b").unwrap();
                wait_for_release(&second_gate);
                Ok(())
            }),
            PretrainedDownloadStatus::Started
        );

        let entered: HashSet<_> = [
            worker_entered.recv_timeout(TIMEOUT).unwrap(),
            worker_entered.recv_timeout(TIMEOUT).unwrap(),
        ]
        .into_iter()
        .collect();
        assert_eq!(entered, HashSet::from(["model-a", "model-b"]));

        release(&release_gate);
        assert!(coordinator.wait_until_idle("model-a", TIMEOUT));
        assert!(coordinator.wait_until_idle("model-b", TIMEOUT));
    }

    #[test]
    fn artifact_aliases_share_one_in_flight_worker() {
        let cases = [
            ("qwen3-asr-0.6b-antirez", "qwen3-asr-antirez"),
            ("qwen3-asr-antirez", "qwen3-asr-0.6b-antirez"),
            ("qwen3-asr-0.6b-ggml", "qwen3-asr-0.6b-ggml-f16"),
            ("qwen3-asr-0.6b-ggml-f16", "qwen3-asr-0.6b-ggml"),
        ];

        for (first_name, alias_name) in cases {
            let coordinator = DownloadCoordinator::default();
            let worker_count = Arc::new(AtomicUsize::new(0));
            let release_gate: ReleaseGate = Arc::new((Mutex::new(false), Condvar::new()));
            let (worker_started, worker_entered) = mpsc::sync_channel(1);

            let first_count = Arc::clone(&worker_count);
            let first_gate = Arc::clone(&release_gate);
            assert_eq!(
                coordinator.spawn_model(first_name, move || {
                    first_count.fetch_add(1, Ordering::SeqCst);
                    worker_started.send(()).unwrap();
                    wait_for_release(&first_gate);
                    Ok(())
                }),
                PretrainedDownloadStatus::Started
            );
            worker_entered.recv_timeout(TIMEOUT).unwrap();

            let alias_count = Arc::clone(&worker_count);
            assert_eq!(
                coordinator.spawn_model(alias_name, move || {
                    alias_count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
                PretrainedDownloadStatus::AlreadyInProgress,
                "{first_name} and {alias_name} must share one artifact key"
            );
            assert_eq!(worker_count.load(Ordering::SeqCst), 1);

            release(&release_gate);
            assert!(coordinator.wait_until_idle(pretrained_download_key(first_name), TIMEOUT));
        }
    }

    #[test]
    fn successful_completion_allows_retry() {
        let coordinator = DownloadCoordinator::default();
        let runs = Arc::new(AtomicUsize::new(0));

        for expected_runs in 1..=2 {
            let worker_runs = Arc::clone(&runs);
            assert_eq!(
                coordinator.spawn("retry-success".to_string(), move || {
                    worker_runs.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
                PretrainedDownloadStatus::Started
            );
            assert!(coordinator.wait_until_idle("retry-success", TIMEOUT));
            assert_eq!(runs.load(Ordering::SeqCst), expected_runs);
        }
    }

    #[test]
    fn failed_completion_allows_retry() {
        let coordinator = DownloadCoordinator::default();
        let runs = Arc::new(AtomicUsize::new(0));

        let failed_runs = Arc::clone(&runs);
        assert_eq!(
            coordinator.spawn("retry-failure".to_string(), move || {
                failed_runs.fetch_add(1, Ordering::SeqCst);
                Err(Error::Download("expected test failure".to_string()))
            }),
            PretrainedDownloadStatus::Started
        );
        assert!(coordinator.wait_until_idle("retry-failure", TIMEOUT));

        let successful_runs = Arc::clone(&runs);
        assert_eq!(
            coordinator.spawn("retry-failure".to_string(), move || {
                successful_runs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }),
            PretrainedDownloadStatus::Started
        );
        assert!(coordinator.wait_until_idle("retry-failure", TIMEOUT));
        assert_eq!(runs.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn panicking_worker_allows_retry() {
        let coordinator = DownloadCoordinator::default();
        let runs = Arc::new(AtomicUsize::new(0));

        let panicking_runs = Arc::clone(&runs);
        assert_eq!(
            coordinator.spawn("retry-panic".to_string(), move || {
                panicking_runs.fetch_add(1, Ordering::SeqCst);
                panic!("expected test panic");
            }),
            PretrainedDownloadStatus::Started
        );
        assert!(coordinator.wait_until_idle("retry-panic", TIMEOUT));

        let successful_runs = Arc::clone(&runs);
        assert_eq!(
            coordinator.spawn("retry-panic".to_string(), move || {
                successful_runs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }),
            PretrainedDownloadStatus::Started
        );
        assert!(coordinator.wait_until_idle("retry-panic", TIMEOUT));
        assert_eq!(runs.load(Ordering::SeqCst), 2);
    }
}

#[cfg(test)]
mod provider_fallback_tests {
    use super::*;

    struct FailingGpu {
        calls: usize,
        fallback_calls: usize,
        cpu_succeeds: bool,
        cpu_initializes: bool,
    }

    impl Engine for FailingGpu {
        fn transcribe(
            &mut self,
            audio: &[f32],
            _sample_rate: u32,
            _opts: &TranscribeOptions,
        ) -> Result<TranscribeResult> {
            self.calls += 1;
            if self.fallback_calls == 0 || !self.cpu_succeeds {
                return Err(Error::Other(if self.fallback_calls == 0 {
                    "representative DirectML device-removed failure".to_string()
                } else {
                    "representative CPU failure".to_string()
                }));
            }
            Ok(TranscribeResult {
                text: format!("recovered {} samples", audio.len()),
                segments: Vec::new(),
            })
        }

        fn name(&self) -> &str {
            "test-parakeet"
        }

        fn fallback_to_cpu(&mut self) -> Result<bool> {
            self.fallback_calls += 1;
            if !self.cpu_initializes {
                return if self.fallback_calls == 1 {
                    Err(Error::Other(
                        "representative CPU construction failure".to_string(),
                    ))
                } else {
                    Ok(false)
                };
            }
            Ok(self.fallback_calls == 1)
        }
    }

    #[test]
    fn gpu_inference_failure_retries_same_audio_once_on_cpu() {
        let engine = FailingGpu {
            calls: 0,
            fallback_calls: 0,
            cpu_succeeds: true,
            cpu_initializes: true,
        };
        let mut model = Model {
            inner: Box::new(engine),
            uses_gpu: false,
        };
        let audio = [0.1, 0.2, 0.3];
        let result = model
            .transcribe(&audio, TranscribeOptions::default())
            .unwrap();
        assert_eq!(result.text, "recovered 3 samples");
    }

    #[test]
    fn gpu_and_cpu_causes_are_preserved_without_a_retry_loop() {
        let engine = FailingGpu {
            calls: 0,
            fallback_calls: 0,
            cpu_succeeds: false,
            cpu_initializes: true,
        };
        let mut model = Model {
            inner: Box::new(engine),
            uses_gpu: false,
        };
        let error = model
            .transcribe(&[0.1], TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("DirectML device-removed"));
        assert!(error.contains("CPU retry failed"));
        assert!(error.contains("representative CPU failure"));
    }

    #[test]
    fn cpu_initialization_failure_preserves_both_causes_and_is_not_retried() {
        let engine = FailingGpu {
            calls: 0,
            fallback_calls: 0,
            cpu_succeeds: false,
            cpu_initializes: false,
        };
        let mut model = Model {
            inner: Box::new(engine),
            uses_gpu: false,
        };
        let first = model
            .transcribe(&[0.1], TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(first.contains("DirectML device-removed"));
        assert!(first.contains("CPU recovery initialization failed"));
        assert!(first.contains("CPU construction failure"));

        let second = model
            .transcribe(&[0.1], TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(second.contains("representative CPU failure"));
    }
}
