// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com

use crate::audio::{self, MelConfig};
use crate::context_bias::{argmax, ContextBiaser};
use crate::error::{Error, Result};
use crate::hf_cache;
use crate::model::{
    Engine, ParakeetExecutionProvider, Segment, TranscribeOptions, TranscribeResult,
};
use ndarray::{Array1, Array2, Array3};
use std::path::{Path, PathBuf};

const DIRECTML_SESSION_INFERENCE_BUDGET: u32 = 512;

/// Parakeet TDT engine using ONNX Runtime.
pub struct ParakeetEngine {
    encoder: Option<ort::session::Session>,
    decoder: Option<ort::session::Session>,
    vocab: Vec<String>,
    vocab_size: usize,
    name: String,
    encoder_path: PathBuf,
    decoder_path: PathBuf,
    provider: ParakeetExecutionProvider,
    recovery_failure: Option<String>,
    initialization_recovery_pending: bool,
    directml_inferences: u32,
}

impl ParakeetEngine {
    /// Load from HuggingFace model name.
    pub fn from_pretrained(name: &str) -> Result<Self> {
        Self::from_pretrained_with_provider(name, ParakeetExecutionProvider::legacy_default())
    }

    pub fn from_pretrained_with_provider(
        name: &str,
        provider: ParakeetExecutionProvider,
    ) -> Result<Self> {
        let model_dir = Self::download_pretrained_files(name)?;

        Self::from_dir_with_provider(&model_dir, provider).map(|mut engine| {
            engine.name = name.to_string();
            engine
        })
    }

    /// Populate the Hugging Face cache without constructing ONNX Runtime sessions.
    pub(crate) fn download_pretrained(name: &str) -> Result<()> {
        Self::download_pretrained_files(name).map(drop)
    }

    fn download_pretrained_files(name: &str) -> Result<PathBuf> {
        let repo = match name {
            "parakeet-tdt-0.6b-v2" => "istupakov/parakeet-tdt-0.6b-v2-onnx",
            "parakeet-tdt-0.6b-v3" => "istupakov/parakeet-tdt-0.6b-v3-onnx",
            other => return Err(Error::ModelNotFound(other.to_string())),
        };

        tracing::info!("downloading {} from {}", name, repo);
        let api = hf_hub::api::sync::Api::new().map_err(|e| Error::Download(e.to_string()))?;
        let model = api.model(repo.to_string());

        // Download int8 models first (preferred: self-contained, CoreML compatible)
        let encoder_file = hf_get_with_retry(&model, "encoder-model.int8.onnx", 3)
            .or_else(|_| hf_get_with_retry(&model, "encoder-model.onnx", 3))
            .map_err(|e| Error::Download(format!("encoder: {e}")))?;
        let model_dir = encoder_file
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        // Download all required files (try int8 first, fallback to fp32)
        for (int8, fp32) in &[("decoder_joint-model.int8.onnx", "decoder_joint-model.onnx")] {
            let _ = hf_get_with_retry(&model, int8, 3)
                .or_else(|_| hf_get_with_retry(&model, fp32, 3))
                .map_err(|e| Error::Download(format!("{fp32}: {e}")))?;
        }
        hf_get_with_retry(&model, "vocab.txt", 3)
            .map_err(|e| Error::Download(format!("vocab.txt: {e}")))?;
        // Try external weights (needed for fp32 encoder, doesn't exist for int8)
        let _ = model.get("encoder-model.onnx.data");

        Ok(model_dir)
    }

    /// Load from HuggingFace cache only — never downloads. Fails with [`Error::ModelNotCached`]
    /// if any required file is missing locally.
    pub fn from_pretrained_cache_only(name: &str) -> Result<Self> {
        Self::from_pretrained_cache_only_with_provider(
            name,
            ParakeetExecutionProvider::legacy_default(),
        )
    }

    pub fn from_pretrained_cache_only_with_provider(
        name: &str,
        provider: ParakeetExecutionProvider,
    ) -> Result<Self> {
        let repo = match name {
            "parakeet-tdt-0.6b-v2" => "istupakov/parakeet-tdt-0.6b-v2-onnx",
            "parakeet-tdt-0.6b-v3" => "istupakov/parakeet-tdt-0.6b-v3-onnx",
            other => return Err(Error::ModelNotFound(other.to_string())),
        };

        let encoder_file = hf_cache::cache_get(repo, "encoder-model.int8.onnx")
            .or_else(|| hf_cache::cache_get(repo, "encoder-model.onnx"))
            .ok_or_else(|| Error::ModelNotCached(name.to_string()))?;
        let model_dir = encoder_file
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        for (int8, fp32) in &[("decoder_joint-model.int8.onnx", "decoder_joint-model.onnx")] {
            let ok = hf_cache::cache_get(repo, int8).is_some()
                || hf_cache::cache_get(repo, fp32).is_some();
            if !ok {
                return Err(Error::ModelNotCached(name.to_string()));
            }
        }
        if hf_cache::cache_get(repo, "vocab.txt").is_none() {
            return Err(Error::ModelNotCached(name.to_string()));
        }
        let _ = hf_cache::cache_get(repo, "encoder-model.onnx.data");

        Self::from_dir_with_provider(&model_dir, provider).map(|mut e| {
            e.name = name.to_string();
            e
        })
    }

    /// Load from a local directory containing ONNX files + vocab.txt.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_with_provider(dir, ParakeetExecutionProvider::legacy_default())
    }

    pub fn from_dir_with_provider(
        dir: &Path,
        requested_provider: ParakeetExecutionProvider,
    ) -> Result<Self> {
        // Prefer int8 models: they're self-contained (no external .data files) and
        // work with CoreML's MLProgram format for ANE acceleration.
        // fp32 models with external data can't use CoreML due to ort limitations.
        let encoder_path = find_file(dir, &["encoder-model.int8.onnx", "encoder-model.onnx"])?;
        let decoder_path = find_file(
            dir,
            &["decoder_joint-model.int8.onnx", "decoder_joint-model.onnx"],
        )?;
        let vocab_path = find_file(dir, &["vocab.txt"])?;

        let vocab = load_vocab(&vocab_path)?;
        let vocab_size = vocab.len();

        let ((encoder, decoder, provider), initialization_recovery_pending) =
            initialize_with_fallback(
                requested_provider,
                || build_sessions(&encoder_path, &decoder_path, requested_provider),
                || build_sessions(&encoder_path, &decoder_path, ParakeetExecutionProvider::Cpu),
            )?;

        tracing::info!("parakeet loaded: vocab_size={}", vocab_size);

        Ok(Self {
            encoder: Some(encoder),
            decoder: Some(decoder),
            vocab,
            vocab_size,
            name: "parakeet".to_string(),
            encoder_path,
            decoder_path,
            provider,
            recovery_failure: None,
            initialization_recovery_pending,
            directml_inferences: 0,
        })
    }

    fn recycle_directml_sessions_if_needed(&mut self) -> Result<()> {
        if self.provider == ParakeetExecutionProvider::Cpu
            || self.directml_inferences < DIRECTML_SESSION_INFERENCE_BUDGET
        {
            return Ok(());
        }

        self.encoder.take();
        self.decoder.take();
        match build_sessions(&self.encoder_path, &self.decoder_path, self.provider) {
            Ok((encoder, decoder, provider)) => {
                self.encoder = Some(encoder);
                self.decoder = Some(decoder);
                self.provider = provider;
                self.directml_inferences = 0;
                tracing::warn!(
                    "parakeet: recycled DirectML sessions after {} inferences to avoid slowdown in long sessions",
                    DIRECTML_SESSION_INFERENCE_BUDGET
                );
                Ok(())
            }
            Err(gpu_error) => {
                tracing::warn!(
                    "parakeet: DirectML session recycle failed ({gpu_error}); falling back to CPU"
                );
                match self.fallback_to_cpu() {
                    Ok(true) => {
                        self.initialization_recovery_pending = true;
                        tracing::warn!(
                            "parakeet: CPU recovery initialized after DirectML session recycle failure ({gpu_error})"
                        );
                        Ok(())
                    }
                    Ok(false) => Err(gpu_error),
                    Err(cpu_error) => Err(Error::Other(format!(
                        "DirectML session recycle failed ({gpu_error}); CPU recovery initialization failed ({cpu_error})"
                    ))),
                }
            }
        }
    }
}

fn initialize_with_fallback<T>(
    requested_provider: ParakeetExecutionProvider,
    directml: impl FnOnce() -> Result<T>,
    cpu: impl FnOnce() -> Result<T>,
) -> Result<(T, bool)> {
    if requested_provider == ParakeetExecutionProvider::Cpu {
        return cpu().map(|value| (value, false));
    }
    match directml() {
        Ok(value) => Ok((value, false)),
        Err(error) => {
            tracing::warn!(
                "parakeet: DirectML initialization failed ({}); falling back to CPU",
                error
            );
            match cpu() {
                Ok(value) => {
                    tracing::warn!(
                        "parakeet: CPU initialization completed after DirectML initialization failure ({error})"
                    );
                    Ok((value, true))
                }
                Err(cpu_error) => Err(Error::Other(format!(
                    "DirectML initialization failed ({error}); CPU initialization failed ({cpu_error})"
                ))),
            }
        }
    }
}

#[cfg(test)]
mod provider_tests {
    use super::*;

    #[test]
    fn directml_initialization_failure_uses_cpu_once() {
        let mut directml_calls = 0;
        let mut cpu_calls = 0;
        let (provider, recovered) = initialize_with_fallback(
            ParakeetExecutionProvider::DirectMl,
            || {
                directml_calls += 1;
                Err(Error::Other(
                    "representative provider registration failure".to_string(),
                ))
            },
            || {
                cpu_calls += 1;
                Ok(ParakeetExecutionProvider::Cpu)
            },
        )
        .unwrap();
        assert_eq!(provider, ParakeetExecutionProvider::Cpu);
        assert!(recovered);
        assert_eq!(directml_calls, 1);
        assert_eq!(cpu_calls, 1);
    }

    #[test]
    fn initialization_failure_preserves_directml_and_cpu_causes() {
        let error = initialize_with_fallback::<()>(
            ParakeetExecutionProvider::DirectMlDevice(99),
            || Err(Error::Other("provider device failure".to_string())),
            || Err(Error::Other("CPU construction failure".to_string())),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("provider device failure"));
        assert!(error.contains("CPU construction failure"));
    }

    #[test]
    #[cfg(all(target_os = "windows", feature = "directml"))]
    #[ignore = "requires the cached real Parakeet model and native DirectML runtime"]
    fn native_directml_recycle_failure_recovers_once_and_reports_completion() {
        use tracing_subscriber::layer::SubscriberExt;

        let filter = std::env::var("SCREENPIPE_TEST_LOG_FILTER")
            .expect("pass the consumer's production log filter")
            .parse::<tracing_subscriber::filter::Targets>()
            .expect("the production filter uses target and level directives");
        let log_path = std::env::var("SCREENPIPE_TEST_DIRECTML_RECOVERY_LOG")
            .expect("pass a private output path for the real support collector");
        let subscriber = tracing_subscriber::registry().with(filter).with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(std::fs::File::create(&log_path).unwrap()),
        );

        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!("native recovery fixture contact=private.person@example.com");
            let mut engine = ParakeetEngine::from_pretrained_cache_only_with_provider(
                "parakeet-tdt-0.6b-v3",
                ParakeetExecutionProvider::Cpu,
            )
            .unwrap();
            engine.provider = ParakeetExecutionProvider::DirectMlDevice(i32::MAX);
            engine.directml_inferences = DIRECTML_SESSION_INFERENCE_BUDGET - 1;
            engine.recycle_directml_sessions_if_needed().unwrap();
            assert_eq!(
                engine.provider,
                ParakeetExecutionProvider::DirectMlDevice(i32::MAX)
            );
            assert_eq!(
                engine.directml_inferences,
                DIRECTML_SESSION_INFERENCE_BUDGET - 1
            );

            engine.directml_inferences = DIRECTML_SESSION_INFERENCE_BUDGET;
            let audio = vec![0.0; 16_000];
            engine
                .transcribe(&audio, 16_000, &TranscribeOptions::default())
                .unwrap();
            assert_eq!(engine.provider, ParakeetExecutionProvider::Cpu);
            assert!(!engine.initialization_recovery_pending);
            assert!(engine.recovery_failure.is_none());
            engine
                .transcribe(&audio, 16_000, &TranscribeOptions::default())
                .unwrap();
            assert_eq!(engine.provider, ParakeetExecutionProvider::Cpu);
        });

        let logs = std::fs::read_to_string(log_path).unwrap();
        assert_eq!(logs.matches("DirectML session recycle failed (").count(), 1);
        assert!(logs.contains("887A0002"));
        assert!(logs.contains("CPU recovery initialized after DirectML session recycle failure"));
        assert!(logs.contains("CPU inference completed after DirectML initialization recovery"));
        assert!(!logs.contains("parakeet: loading"));
    }

    #[test]
    #[cfg(all(target_os = "windows", feature = "directml"))]
    #[ignore = "requires the cached real Parakeet model and native DirectML runtime"]
    fn native_directml_recycle_failed_cpu_initialization_is_sticky() {
        let mut engine = ParakeetEngine::from_pretrained_cache_only_with_provider(
            "parakeet-tdt-0.6b-v3",
            ParakeetExecutionProvider::Cpu,
        )
        .unwrap();
        engine.provider = ParakeetExecutionProvider::DirectMlDevice(i32::MAX);
        engine.directml_inferences = DIRECTML_SESSION_INFERENCE_BUDGET;
        engine.encoder_path = engine.encoder_path.join("missing-recovery-model.onnx");
        let audio = vec![0.0; 16_000];
        let first = engine
            .transcribe(&audio, 16_000, &TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(first.contains("DirectML session recycle failed"));
        assert!(first.contains("CPU recovery initialization failed"));
        assert_eq!(engine.provider, ParakeetExecutionProvider::Cpu);
        assert!(engine.encoder.is_none() && engine.decoder.is_none());
        let saved_failure = engine.recovery_failure.clone();
        assert!(saved_failure.is_some());
        let second = engine
            .transcribe(&audio, 16_000, &TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(second.contains("CPU recovery is unavailable"));
        assert_eq!(engine.recovery_failure, saved_failure);
    }
}

/// Build an ONNX session with the best available execution provider.
///
/// Parakeet uses CPU execution on macOS — benchmarks show it's 2x faster and
/// uses 4x less memory than CoreML on Apple Silicon, because only ~44% of
/// encoder ops are CoreML-compatible, and the CPU↔ANE data transfer overhead
/// negates the acceleration benefit.
///
/// On Windows with `directml` feature: tries DirectML for GPU acceleration.
/// Falls back to CPU if no accelerator works.
fn build_sessions(
    encoder_path: &Path,
    decoder_path: &Path,
    provider: ParakeetExecutionProvider,
) -> Result<(
    ort::session::Session,
    ort::session::Session,
    ParakeetExecutionProvider,
)> {
    tracing::info!("loading encoder from {}", encoder_path.display());
    let encoder = build_session_with_ep(encoder_path, provider)?;
    tracing::info!("loading decoder from {}", decoder_path.display());
    let decoder = build_session_with_ep(decoder_path, provider)?;
    Ok((encoder, decoder, provider))
}

fn build_session_with_ep(
    onnx_path: &std::path::Path,
    provider: ParakeetExecutionProvider,
) -> Result<ort::session::Session> {
    let file_name = onnx_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    #[cfg(feature = "directml")]
    {
        if provider != ParakeetExecutionProvider::Cpu {
            // ort rc.12: each builder step returns Result<SessionBuilder, Error<SessionBuilder>>,
            // whose error type doesn't unify across `.and_then` (E0308), so normalize each step
            // with map_err + `?` exactly like the CPU path below. Wrapped in a closure so the
            // CPU fallback can still match on Err.
            let directml_session: Result<ort::session::Session> = (|| {
                Ok(ort::session::Session::builder()?
                    .with_execution_providers([directml_provider(provider)])
                    .map_err(|e| Error::Other(e.to_string()))?
                    .with_parallel_execution(false)
                    .map_err(|e| Error::Other(e.to_string()))?
                    .with_memory_pattern(false)
                    .map_err(|e| Error::Other(e.to_string()))?
                    .commit_from_file(onnx_path)?)
            })();
            match directml_session {
                Ok(session) => {
                    tracing::info!("parakeet: DirectML session created for {}", file_name);
                    return Ok(session);
                }
                Err(e) => {
                    return Err(Error::Other(format!(
                        "DirectML failed for {file_name}: {e}"
                    )));
                }
            }
        }
    }

    #[cfg(not(feature = "directml"))]
    if provider != ParakeetExecutionProvider::Cpu {
        return Err(Error::Other(
            "DirectML support is not compiled in".to_string(),
        ));
    }

    // CPU execution with thread limiting — default path.
    let intra_threads = 1;
    tracing::info!(
        "parakeet: loading {} on CPU ({} threads)",
        file_name,
        intra_threads
    );
    Ok(ort::session::Session::builder()?
        .with_intra_threads(intra_threads)
        .map_err(|e| Error::Other(e.to_string()))?
        .with_inter_threads(1)
        .map_err(|e| Error::Other(e.to_string()))?
        .commit_from_file(onnx_path)?)
}

#[cfg(feature = "directml")]
fn directml_provider(
    provider: ParakeetExecutionProvider,
) -> ort::execution_providers::ExecutionProviderDispatch {
    let directml = ort::execution_providers::DirectMLExecutionProvider::default();
    match provider {
        ParakeetExecutionProvider::DirectMlDevice(device_id) => directml
            .with_device_id(device_id)
            .build()
            .error_on_failure(),
        ParakeetExecutionProvider::DirectMl => directml
            .with_performance_preference(
                ort::execution_providers::directml::PerformancePreference::HighPerformance,
            )
            .build()
            .error_on_failure(),
        ParakeetExecutionProvider::Cpu => unreachable!("CPU does not use DirectML"),
    }
}

/// Helper to extract f32 tensor from ort output as a raw shape + data.
fn extract_f32(val: &ort::value::DynValue) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = val
        .try_extract_array::<f32>()
        .map_err(|e| Error::Other(format!("extract tensor: {e}")))?;
    let dims: Vec<usize> = view.shape().to_vec();
    Ok((dims, view.iter().copied().collect()))
}

impl Engine for ParakeetEngine {
    fn transcribe(
        &mut self,
        audio_samples: &[f32],
        sample_rate: u32,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        if let Some(error) = &self.recovery_failure {
            return Err(Error::Other(format!(
                "Parakeet CPU recovery is unavailable after GPU failure: {error}"
            )));
        }
        self.recycle_directml_sessions_if_needed()?;
        let encoder = self.encoder.as_mut().ok_or_else(|| {
            Error::Other("Parakeet encoder unavailable after provider transition".to_string())
        })?;
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            Error::Other("Parakeet decoder unavailable after provider transition".to_string())
        })?;
        let audio = if sample_rate != 16000 {
            audio::resample(audio_samples, sample_rate, 16000)
        } else {
            audio_samples.to_vec()
        };

        // Extract mel features [time x n_mels]
        let features = audio::mel_spectrogram(&audio, &MelConfig::nemo());
        let n_frames = features.shape()[0];
        let n_feats = features.shape()[1];

        // Encoder expects [batch=1, features, time]
        let input = features
            .t()
            .to_shape((1, n_feats, n_frames))
            .map_err(|e| Error::Other(format!("reshape: {e}")))?
            .to_owned();
        let input_len = Array1::from_vec(vec![n_frames as i64]);

        let enc_inputs = ort::inputs![
            "audio_signal" => ort::value::TensorRef::from_array_view(input.view())?,
            "length" => ort::value::TensorRef::from_array_view(input_len.view())?
        ];
        let enc_out = encoder.run(enc_inputs)?;

        let (enc_shape, enc_data) = extract_f32(&enc_out["outputs"])?;
        let (b, enc_t, enc_d) = (enc_shape[0], enc_shape[1], enc_shape[2]);
        let enc_array = Array3::from_shape_vec((b, enc_t, enc_d), enc_data)
            .map_err(|e| Error::Other(format!("encoder array: {e}")))?;

        // Greedy TDT decode, with optional contextual biasing toward keyterms.
        let biaser = ContextBiaser::build(&opts.keyterms, &self.vocab, opts.keyterm_boost);
        let (tokens, frame_indices) =
            greedy_tdt_decode(decoder, &enc_array, self.vocab_size, biaser.as_ref())?;

        // Tokens to text
        let audio_secs = audio.len() as f64 / 16000.0;
        let frame_rate = if enc_t > 0 {
            audio_secs / enc_t as f64
        } else {
            0.0
        };
        let mut text = String::new();
        let mut segments = Vec::new();

        for (i, &tok) in tokens.iter().enumerate() {
            if tok < self.vocab.len() {
                let token_text = &self.vocab[tok];
                let word = token_text.replace('▁', " ");
                text.push_str(&word);

                let start = frame_indices[i] as f64 * frame_rate;
                segments.push(Segment {
                    start_secs: start,
                    end_secs: start + frame_rate,
                    text: word,
                });
            }
        }

        let result = TranscribeResult {
            text: text.trim().to_string(),
            segments,
        };
        if self.initialization_recovery_pending {
            tracing::warn!(
                "parakeet: CPU inference completed after DirectML initialization recovery"
            );
            self.initialization_recovery_pending = false;
        }
        if self.provider != ParakeetExecutionProvider::Cpu {
            self.directml_inferences = self.directml_inferences.saturating_add(1);
        }
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn execution_provider(&self) -> Option<&'static str> {
        Some(match self.provider {
            ParakeetExecutionProvider::Cpu => "CPU",
            ParakeetExecutionProvider::DirectMl | ParakeetExecutionProvider::DirectMlDevice(_) => {
                "DirectML"
            }
        })
    }

    fn fallback_to_cpu(&mut self) -> Result<bool> {
        if self.provider == ParakeetExecutionProvider::Cpu {
            return Ok(false);
        }
        self.encoder.take();
        self.decoder.take();
        self.provider = ParakeetExecutionProvider::Cpu;
        let (encoder, decoder, provider) = match build_sessions(
            &self.encoder_path,
            &self.decoder_path,
            ParakeetExecutionProvider::Cpu,
        ) {
            Ok(sessions) => sessions,
            Err(error) => {
                self.recovery_failure = Some(error.to_string());
                return Err(error);
            }
        };
        self.encoder = Some(encoder);
        self.decoder = Some(decoder);
        self.provider = provider;
        Ok(true)
    }
}

/// Frame-by-frame TDT greedy decode.
fn greedy_tdt_decode(
    decoder: &mut ort::session::Session,
    encoder_out: &Array3<f32>,
    vocab_size: usize,
    biaser: Option<&ContextBiaser>,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let enc_dim = encoder_out.shape()[1];
    let time_steps = encoder_out.shape()[2];
    let blank_id = vocab_size - 1;
    let max_tokens_per_step = 10;

    let mut state_h = Array3::<f32>::zeros((2, 1, 640));
    let mut state_c = Array3::<f32>::zeros((2, 1, 640));

    let mut tokens = Vec::new();
    let mut frame_indices = Vec::new();
    let mut t = 0;
    let mut emitted = 0;
    let mut last_token = blank_id as i32;
    // Active keyterm-trie nodes for contextual biasing; root (0) is always
    // active so a phrase can begin at any emission. Unused when biaser is None.
    let mut active: Vec<usize> = vec![0];

    while t < time_steps {
        let frame = encoder_out.slice(ndarray::s![0, .., t]).to_owned();
        let frame = frame
            .to_shape((1, enc_dim, 1))
            .map_err(|e| Error::Other(format!("frame reshape: {e}")))?
            .to_owned();

        let targets = Array2::from_shape_vec((1, 1), vec![last_token])
            .map_err(|e| Error::Other(format!("targets: {e}")))?;

        let target_length = Array1::from_vec(vec![1i32]);
        let dec_inputs = ort::inputs![
            "encoder_outputs" => ort::value::TensorRef::from_array_view(frame.view())?,
            "targets" => ort::value::TensorRef::from_array_view(targets.view())?,
            "target_length" => ort::value::TensorRef::from_array_view(target_length.view())?,
            "input_states_1" => ort::value::TensorRef::from_array_view(state_h.view())?,
            "input_states_2" => ort::value::TensorRef::from_array_view(state_c.view())?
        ];
        let out = decoder.run(dec_inputs)?;

        let (_, logits) = extract_f32(&out["outputs"])?;

        let token_id = match biaser {
            // Shallow fusion: boost tokens that advance an active keyterm match,
            // then argmax over the biased vocab logits.
            Some(b) => {
                let mut scores: Vec<f32> = logits[..vocab_size].to_vec();
                b.apply(&active, &mut scores);
                argmax(&scores).unwrap_or(blank_id)
            }
            None => logits
                .iter()
                .take(vocab_size)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(blank_id),
        };

        let dur = logits
            .iter()
            .skip(vocab_size)
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        if token_id != blank_id {
            if let Ok((sh, sd)) = extract_f32(&out["output_states_1"]) {
                if let Ok(arr) = Array3::from_shape_vec((sh[0], sh[1], sh[2]), sd) {
                    state_h = arr;
                }
            }
            if let Ok((sh, sd)) = extract_f32(&out["output_states_2"]) {
                if let Ok(arr) = Array3::from_shape_vec((sh[0], sh[1], sh[2]), sd) {
                    state_c = arr;
                }
            }
            tokens.push(token_id);
            frame_indices.push(t);
            last_token = token_id as i32;
            emitted += 1;
            if let Some(b) = biaser {
                active = b.advance(&active, token_id);
            }
        }

        if dur > 0 {
            t += dur;
            emitted = 0;
        } else if token_id == blank_id || emitted >= max_tokens_per_step {
            t += 1;
            emitted = 0;
        }
    }

    Ok((tokens, frame_indices))
}

/// Download a file from HuggingFace with retry on transient connection errors.
fn hf_get_with_retry(
    model: &hf_hub::api::sync::ApiRepo,
    filename: &str,
    max_retries: u32,
) -> std::result::Result<PathBuf, String> {
    let mut last_err = String::new();
    for attempt in 0..max_retries {
        match model.get(filename) {
            Ok(path) => return Ok(path),
            Err(e) => {
                last_err = e.to_string();
                if attempt + 1 < max_retries {
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                    tracing::warn!(
                        "parakeet: download {} failed (attempt {}/{}): {}, retrying in {:?}",
                        filename,
                        attempt + 1,
                        max_retries,
                        last_err,
                        delay
                    );
                    std::thread::sleep(delay);
                }
            }
        }
    }
    Err(last_err)
}

fn find_file(dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
    for c in candidates {
        let p = dir.join(c);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(Error::ModelNotFound(format!(
        "none of {:?} found in {}",
        candidates,
        dir.display()
    )))
}

fn load_vocab(path: &Path) -> Result<Vec<String>> {
    let content =
        std::fs::read_to_string(path).map_err(|e| Error::Other(format!("read vocab: {e}")))?;
    let mut vocab = Vec::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() == 2 {
            let token = parts[0].to_string();
            let id: usize = parts[1]
                .parse()
                .map_err(|e| Error::Other(format!("vocab id: {e}")))?;
            if id >= vocab.len() {
                vocab.resize(id + 1, String::new());
            }
            vocab[id] = token;
        }
    }
    Ok(vocab)
}
