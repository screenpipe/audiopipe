// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
//! Native Bark weights with ONNX Runtime CPU / Windows DirectML sessions.
use super::onnx_weights::{self, Initializer};
use crate::{
    error::{Error, Result},
    model::{Engine, ParakeetExecutionProvider},
    TranscribeOptions, TranscribeResult,
};
use ndarray::{Array1, Array3};
use ort::session::{builder::SessionBuilder, Session};
use std::{path::Path, sync::Arc};

pub(crate) struct BarkOnnxEngine {
    encoder: Option<Session>,
    decoder: Option<Session>,
    weights: Arc<Vec<Initializer>>,
    provider: ParakeetExecutionProvider,
    vocab: Vec<String>,
    directml_inferences: u32,
    recovery_failure: Option<String>,
}
impl BarkOnnxEngine {
    pub(crate) fn from_pretrained(provider: ParakeetExecutionProvider) -> Result<Self> {
        let p = super::download().map_err(|e| Error::Download(format!("{e:#}")))?;
        Self::from_path(&p, provider)
    }
    pub(crate) fn from_pretrained_cache_only(provider: ParakeetExecutionProvider) -> Result<Self> {
        let p = super::cached_model()
            .ok_or_else(|| Error::ModelNotCached(super::ONNX_MODEL_NAME.into()))?;
        Self::from_path(&p, provider)
    }
    pub(crate) fn from_dir(path: &Path, provider: ParakeetExecutionProvider) -> Result<Self> {
        Self::from_path(&path.join("model.bark"), provider)
    }
    fn from_path(path: &Path, provider: ParakeetExecutionProvider) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(
                path.parent()
                    .ok_or_else(|| Error::Other("missing model directory".into()))?
                    .join("mlx/config.json"),
            )
            .map_err(|e| Error::Other(e.to_string()))?,
        )
        .map_err(|e| Error::Other(e.to_string()))?;
        let vocab: Vec<String> = serde_json::from_value(config["joint"]["vocabulary"].clone())
            .map_err(|e| Error::Other(e.to_string()))?;
        if vocab.len() != 8192 {
            return Err(Error::Other("unexpected Bark vocabulary".into()));
        }
        let weights = Arc::new(
            onnx_weights::load(path)
                .map_err(|e| Error::Other(format!("Bark ONNX weights: {e:#}")))?,
        );
        let (encoder, decoder, provider) = match sessions(&weights, provider) {
            Ok((e, d)) => (e, d, provider),
            Err(error) if provider != ParakeetExecutionProvider::Cpu => {
                tracing::warn!(
                    "Bark DirectML initialization failed ({error}); loading CPU sessions"
                );
                let (e, d) = sessions(&weights, ParakeetExecutionProvider::Cpu).map_err(|cpu_error| Error::Other(format!(
                    "Bark DirectML initialization failed ({error}); CPU initialization failed ({cpu_error})"
                )))?;
                (e, d, ParakeetExecutionProvider::Cpu)
            }
            Err(e) => return Err(e),
        };
        Ok(Self {
            encoder: Some(encoder),
            decoder: Some(decoder),
            weights,
            provider,
            vocab,
            directml_inferences: 0,
            recovery_failure: None,
        })
    }
    fn recycle_if_needed(&mut self) -> Result<()> {
        if self.provider == ParakeetExecutionProvider::Cpu || self.directml_inferences < 512 {
            return Ok(());
        }
        match sessions(&self.weights, self.provider) {
            Ok((e, d)) => {
                self.encoder = Some(e);
                self.decoder = Some(d);
                self.directml_inferences = 0;
                Ok(())
            }
            Err(error) => {
                tracing::warn!("Bark DirectML recycle failed ({error}); using CPU");
                self.fallback_to_cpu().map(|_| ())
            }
        }
    }
}
fn builder(provider: ParakeetExecutionProvider) -> Result<SessionBuilder> {
    let mut b = Session::builder()?
        .with_no_environment_execution_providers()
        .map_err(|e| Error::Other(e.to_string()))?
        .with_parallel_execution(false)
        .map_err(|e| Error::Other(e.to_string()))?
        .with_memory_pattern(false)
        .map_err(|e| Error::Other(e.to_string()))?;
    if provider == ParakeetExecutionProvider::Cpu {
        return b
            .with_intra_threads(2)
            .map_err(|e| Error::Other(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| Error::Other(e.to_string()));
    }
    #[cfg(all(feature = "directml", target_os = "windows"))]
    {
        let dml = ort::execution_providers::DirectMLExecutionProvider::default();
        let ep = match provider {
            ParakeetExecutionProvider::DirectMlDevice(id) => dml.with_device_id(id),
            _ => dml.with_performance_preference(
                ort::execution_providers::directml::PerformancePreference::HighPerformance,
            ),
        };
        b = b
            .with_execution_providers([ep.build().error_on_failure()])
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(b)
    }
    #[cfg(not(all(feature = "directml", target_os = "windows")))]
    {
        let _ = &mut b;
        Err(Error::Other(
            "Bark DirectML requires a Windows build with the directml feature".into(),
        ))
    }
}
fn sessions(
    weights: &[Initializer],
    provider: ParakeetExecutionProvider,
) -> Result<(Session, Session)> {
    let mut e = builder(provider)?;
    let mut d = builder(provider)?;
    for (name, size) in [
        ("audio_signal_dynamic_axes_1", 1),
        (
            "audio_signal_dynamic_axes_2",
            super::onnx_frontend::MAX_FRAMES as i64,
        ),
        ("length_dynamic_axes_1", 1),
    ] {
        e = e
            .with_dimension_override(name, size)
            .map_err(|e| Error::Other(e.to_string()))?;
    }
    for name in [
        "encoder_outputs_dynamic_axes_1",
        "encoder_outputs_dynamic_axes_2",
        "targets_dynamic_axes_1",
        "targets_dynamic_axes_2",
        "target_length_dynamic_axes_1",
        "input_states_1_dynamic_axes_1",
        "input_states_2_dynamic_axes_1",
    ] {
        d = d
            .with_dimension_override(name, 1)
            .map_err(|e| Error::Other(e.to_string()))?;
    }
    for w in weights {
        if w.graph == "encoder" {
            e = e
                .with_external_initializer(&w.name, Arc::clone(&w.value))
                .map_err(|e| Error::Other(e.to_string()))?;
        } else {
            d = d
                .with_external_initializer(&w.name, Arc::clone(&w.value))
                .map_err(|e| Error::Other(e.to_string()))?;
        }
    }
    Ok((
        e.commit_from_memory(include_bytes!("onnx/encoder.onnx"))?,
        d.commit_from_memory(include_bytes!("onnx/decoder.onnx"))?,
    ))
}
impl Engine for BarkOnnxEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        opts: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        if let Some(error) = &self.recovery_failure {
            return Err(Error::Other(format!(
                "Bark CPU recovery unavailable: {error}"
            )));
        }
        if sample_rate == 0 || audio.iter().any(|s| !s.is_finite()) {
            return Err(Error::InvalidAudio(
                "Bark requires finite audio and a positive sample rate".into(),
            ));
        }
        if opts.language.as_deref().is_some_and(|l| l != "en") {
            return Err(Error::Other("Bark supports English only".into()));
        }
        if audio.is_empty() || audio.iter().all(|s| *s == 0.) {
            return Ok(TranscribeResult {
                text: String::new(),
                segments: Vec::new(),
            });
        }
        let audio = crate::audio::resample(audio, sample_rate, 16000);
        let mut texts = Vec::new();
        for chunk in audio.chunks(30 * 16000) {
            self.recycle_if_needed()?;
            let mut padded = vec![0.; 4000];
            padded.extend_from_slice(chunk);
            // Normalize over real samples, then pad the feature tensor to a fixed
            // 30.25s shape so DirectML can reuse allocations without changing audio.
            let (input, length) = super::onnx_frontend::features(&padded)?;
            let lens = Array1::from_vec(vec![length as i64]);
            let output=self.encoder.as_mut().ok_or_else(||Error::Other("Bark encoder unavailable".into()))?.run(ort::inputs!["audio_signal"=>ort::value::TensorRef::from_array_view(input.view())?,"length"=>ort::value::TensorRef::from_array_view(lens.view())?])?;
            let encoded = output["outputs"].try_extract_array::<f32>()?;
            let shape = encoded.shape();
            let lengths = output["encoded_lengths"].try_extract_array::<i64>()?;
            let steps = usize::try_from(lengths.iter().next().copied().unwrap_or(0))
                .map_err(|e| Error::Other(e.to_string()))?;
            if shape.len() != 3 || shape[0] != 1 || shape[1] != 1024 || steps > shape[2] {
                return Err(Error::Decode("unexpected Bark encoder output shape".into()));
            }
            let encoded = Array3::from_shape_vec(
                (shape[0], shape[1], shape[2]),
                encoded.iter().copied().collect(),
            )
            .map_err(|e| Error::Other(e.to_string()))?;
            let encoded = encoded.slice(ndarray::s![.., .., ..steps]).to_owned();
            let (tokens, _) = crate::parakeet::greedy_tdt_decode(
                self.decoder
                    .as_mut()
                    .ok_or_else(|| Error::Other("Bark decoder unavailable".into()))?,
                &encoded,
                8193,
                None,
            )?;
            let mut text = String::new();
            for id in tokens {
                if let Some(piece) = self.vocab.get(id) {
                    if !piece.starts_with('<') {
                        text.push_str(&piece.replace('▁', " "));
                    }
                }
            }
            if !text.trim().is_empty() {
                texts.push(text.trim().to_string());
            }
            if self.provider != ParakeetExecutionProvider::Cpu {
                self.directml_inferences += 1;
            }
        }
        Ok(TranscribeResult {
            text: texts.join(" "),
            segments: Vec::new(),
        })
    }
    fn name(&self) -> &str {
        super::ONNX_MODEL_NAME
    }
    fn execution_provider(&self) -> Option<&'static str> {
        if self.recovery_failure.is_some() {
            return None;
        }
        Some(if self.provider == ParakeetExecutionProvider::Cpu {
            "CPU"
        } else {
            "DirectML"
        })
    }
    fn fallback_to_cpu(&mut self) -> Result<bool> {
        if let Some(error) = &self.recovery_failure {
            return Err(Error::Other(error.clone()));
        }
        if self.provider == ParakeetExecutionProvider::Cpu {
            return Ok(false);
        }
        self.encoder.take();
        self.decoder.take();
        let (e, d) = match sessions(&self.weights, ParakeetExecutionProvider::Cpu) {
            Ok(sessions) => sessions,
            Err(error) => {
                self.recovery_failure = Some(error.to_string());
                return Err(error);
            }
        };
        self.encoder = Some(e);
        self.decoder = Some(d);
        self.provider = ParakeetExecutionProvider::Cpu;
        self.directml_inferences = 0;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_recovery_is_sticky_and_never_claims_cpu_or_gpu_success() {
        let mut engine = BarkOnnxEngine {
            encoder: None,
            decoder: None,
            weights: Arc::new(Vec::new()),
            provider: ParakeetExecutionProvider::DirectMl,
            vocab: Vec::new(),
            directml_inferences: 512,
            recovery_failure: Some("CPU recovery initialization failed".into()),
        };
        assert_eq!(engine.execution_provider(), None);
        assert!(engine
            .transcribe(&[0.5], 16000, &TranscribeOptions::default())
            .is_err());
        assert!(engine
            .fallback_to_cpu()
            .unwrap_err()
            .to_string()
            .contains("CPU recovery initialization failed"));
        assert!(engine.encoder.is_none() && engine.decoder.is_none());
    }
}
