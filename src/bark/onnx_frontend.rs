// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
//! Parakeet/Bark frontend: centered constant-pad STFT, symmetric Hann window,
//! Slaney mel filters, log(energy + 2^-24), sample standard deviation over real
//! frames. Matches the Hugging Face Parakeet processor used by official Bark.
use crate::{error::Result, Error};
use ndarray::{Array2, Array3};
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::{Arc, OnceLock};
const HOP: usize = 160;
pub(super) const MAX_FRAMES: usize = 3026;
struct Frontend {
    window: Vec<f32>,
    filters: Array2<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}
fn frontend() -> &'static Frontend {
    static F: OnceLock<Frontend> = OnceLock::new();
    F.get_or_init(|| Frontend {
        window: (0..400)
            .map(|i| 0.5 - 0.5 * (2. * std::f32::consts::PI * i as f32 / 399.).cos())
            .collect(),
        filters: crate::audio::mel_filterbank(512, 128, 16000),
        fft: RealFftPlanner::new().plan_fft_forward(512),
    })
}
pub(super) fn features(audio: &[f32]) -> Result<(Array3<f32>, usize)> {
    let frames = audio.len() / HOP;
    if !(2..MAX_FRAMES).contains(&frames) {
        return Err(Error::InvalidAudio(
            "Bark feature input outside fixed 30.25 second window".into(),
        ));
    }
    let f = frontend();
    let mut samples = audio.to_vec();
    for i in (1..samples.len()).rev() {
        samples[i] -= 0.97 * samples[i - 1];
    }
    let mut energy = Array2::zeros((257, frames));
    let mut input = f.fft.make_input_vec();
    let mut output = f.fft.make_output_vec();
    let mut scratch = f.fft.make_scratch_vec();
    for t in 0..frames {
        input.fill(0.);
        for i in 0..400 {
            let sample = (t * HOP + 56 + i) as isize - 256;
            if sample >= 0 && (sample as usize) < samples.len() {
                input[56 + i] = samples[sample as usize] * f.window[i];
            }
        }
        f.fft
            .process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| Error::Other(e.to_string()))?;
        for k in 0..257 {
            let magnitude = output[k].norm_sqr().sqrt();
            energy[[k, t]] = magnitude * magnitude;
        }
    }
    let mel = f.filters.dot(&energy).mapv(|v| (v + 2.0f32.powi(-24)).ln());
    let mut out = Array3::zeros((1, 128, MAX_FRAMES));
    for c in 0..128 {
        let row = mel.row(c);
        let mean = row.iter().sum::<f32>() / frames as f32;
        let std = (row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (frames - 1) as f32)
            .sqrt()
            + 1e-5;
        for t in 0..frames {
            out[[0, c, t]] = (row[t] - mean) / std;
        }
    }
    Ok((out, frames))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_independent_transformers_frontend_oracle() {
        let wave: Vec<_> = (0..5600)
            .map(|i| {
                if i < 4000 {
                    0.
                } else {
                    ((i % 97) as f32 - 48.) / 64.
                }
            })
            .collect();
        let (v, n) = features(&wave).unwrap();
        let golden = include_bytes!("../../tests/bark-fixtures/frontend-35.f32");
        let mut max_error = 0f32;
        for c in 0..128 {
            for t in 0..n {
                let offset = (c * n + t) * 4;
                let expected = f32::from_le_bytes(golden[offset..offset + 4].try_into().unwrap());
                max_error = max_error.max((v[[0, c, t]] - expected).abs());
            }
        }
        assert!(max_error < 2e-3, "frontend max error {max_error}");
    }
    #[test]
    fn real_frames_normalized_and_padding_remains_zero() {
        let wave: Vec<_> = (0..5600)
            .map(|i| {
                if i < 4000 {
                    0.
                } else {
                    (i as f32 * 0.17).sin() * 0.1
                }
            })
            .collect();
        let (v, n) = features(&wave).unwrap();
        assert_eq!(n, 35);
        assert_eq!(v.shape(), [1, 128, MAX_FRAMES]);
        for c in 0..128 {
            let mean = (0..n).map(|t| v[[0, c, t]]).sum::<f32>() / n as f32;
            assert!(mean.abs() < 2e-3);
            assert!((n..MAX_FRAMES).all(|t| v[[0, c, t]] == 0.));
        }
    }
}
