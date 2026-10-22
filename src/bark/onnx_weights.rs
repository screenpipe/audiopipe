// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
//! Bind the exact Bark tensors to weight-free ONNX architecture templates.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, Read},
    path::Path,
    sync::Arc,
};

#[derive(Deserialize)]
pub(super) struct Binding {
    pub graph: String,
    pub target: String,
    pub shape: Vec<usize>,
    pub sources: Vec<String>,
    pub transform: String,
}
pub(super) struct Initializer {
    pub graph: String,
    pub name: String,
    pub value: Arc<ort::value::DynValue>,
}

pub(super) fn load(path: &Path) -> Result<Vec<Initializer>> {
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    let mut buf = [0; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == super::MODEL_SHA256,
        "Bark model SHA-256 mismatch"
    );
    let bindings: Vec<Binding> = serde_json::from_str(include_str!("onnx/bindings.json"))?;
    ensure!(bindings.len() == 626, "unexpected ONNX binding count");
    let mut waiting: HashMap<String, Vec<usize>> = HashMap::new();
    let mut refs: HashMap<String, usize> = HashMap::new();
    let mut targets = HashSet::new();
    for (i, b) in bindings.iter().enumerate() {
        ensure!(
            targets.insert((b.graph.clone(), b.target.clone())),
            "duplicate ONNX initializer"
        );
        for source in &b.sources {
            waiting.entry(source.clone()).or_default().push(i);
            *refs.entry(source.clone()).or_default() += 1;
        }
    }
    ensure!(waiting.len() == 699, "unexpected source tensor count");
    let mut pending: HashMap<String, super::weights::Tensor> = HashMap::new();
    let mut result = Vec::new();
    let mut done = HashSet::new();
    super::weights::read(BufReader::new(File::open(path)?), |tensor| {
        let name = tensor.name.clone();
        ensure!(waiting.contains_key(&name), "unexpected Bark tensor {name}");
        ensure!(
            pending.insert(name.clone(), tensor).is_none(),
            "duplicate Bark tensor {name}"
        );
        for &i in &waiting[&name] {
            let b = &bindings[i];
            if done.contains(&i) || !b.sources.iter().all(|s| pending.contains_key(s)) {
                continue;
            }
            let arrays: Vec<_> = b
                .sources
                .iter()
                .map(|s| pending[s].values.as_slice())
                .collect();
            let values = convert(b, &arrays)?;
            result.push(initializer(b, values)?);
            done.insert(i);
            for s in &b.sources {
                let count = refs.get_mut(s).context("missing source reference")?;
                *count -= 1;
                if *count == 0 {
                    pending.remove(s);
                }
            }
        }
        Ok(())
    })?;
    for (i, b) in bindings
        .iter()
        .enumerate()
        .filter(|(_, b)| b.sources.is_empty())
    {
        result.push(initializer(b, convert(b, &[])?)?);
        done.insert(i);
    }
    ensure!(
        done.len() == bindings.len() && pending.is_empty(),
        "incomplete ONNX weight reconstruction"
    );
    Ok(result)
}
fn initializer(b: &Binding, values: Vec<f32>) -> Result<Initializer> {
    ensure!(
        values.len() == b.shape.iter().product::<usize>(),
        "shape mismatch: {}",
        b.target
    );
    let shape: Vec<i64> = b
        .shape
        .iter()
        .map(|&x| i64::try_from(x))
        .collect::<std::result::Result<_, _>>()?;
    let value = ort::value::Tensor::from_array((shape, values.into_boxed_slice()))?.into_dyn();
    Ok(Initializer {
        graph: b.graph.clone(),
        name: b.target.clone(),
        value: Arc::new(value),
    })
}
fn gate_order(values: &[f32]) -> Result<Vec<f32>> {
    ensure!(values.len() % 4 == 0, "invalid LSTM gate shape");
    let stride = values.len() / 4;
    Ok([0, 3, 1, 2]
        .iter()
        .flat_map(|&i| values[i * stride..(i + 1) * stride].iter().copied())
        .collect())
}
fn convert(b: &Binding, a: &[&[f32]]) -> Result<Vec<f32>> {
    let count = b.shape.iter().product::<usize>();
    let values = match b.transform.as_str() {
        "identity" => a[0].to_vec(),
        "transpose" => {
            ensure!(
                b.shape.len() == 2 && a[0].len() == count,
                "invalid matrix shape"
            );
            let (rows, cols) = (b.shape[0], b.shape[1]);
            let mut out = vec![0.; count];
            for i in 0..rows {
                for j in 0..cols {
                    out[i * cols + j] = a[0][j * rows + i];
                }
            }
            out
        }
        "lstm" => gate_order(a[0])?,
        "lstm_bias" => {
            let mut out = gate_order(a[0])?;
            out.extend(gate_order(a[1])?);
            out
        }
        "batch_norm_weight" => {
            ensure!(
                a[0].len() == count && a[1].len() == b.shape[0] && a[2].len() == b.shape[0],
                "invalid BN shape"
            );
            let stride = count / b.shape[0];
            a[0].iter()
                .enumerate()
                .map(|(i, &w)| w * a[1][i / stride] / (a[2][i / stride] + 1e-5).sqrt())
                .collect()
        }
        "batch_norm_bias" => {
            ensure!(a.iter().all(|v| v.len() == count), "invalid BN bias shape");
            (0..count)
                .map(|i| a[0][i] - a[2][i] * a[1][i] / (a[3][i] + 1e-5).sqrt())
                .collect()
        }
        "position" => {
            ensure!(b.shape == [1, 9999, 1024], "invalid positional shape");
            let mut out = Vec::with_capacity(count);
            let div: Vec<f32> = (0..512)
                .map(|i| ((i * 2) as f32 * (-10000f32.ln() / 1024.)).exp())
                .collect();
            for p in (0..9999).map(|i| 4999. - i as f32) {
                for &d in &div {
                    let x = p * d;
                    out.push(x.sin());
                    out.push(x.cos());
                }
            }
            out
        }
        other => anyhow::bail!("unknown weight transform {other}"),
    };
    ensure!(
        values.len() == count,
        "converted shape mismatch: {}",
        b.target
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn binding(shape: &[usize], transform: &str) -> Binding {
        Binding {
            graph: "decoder".into(),
            target: "test".into(),
            shape: shape.to_vec(),
            sources: vec![],
            transform: transform.into(),
        }
    }
    #[test]
    fn lstm_pytorch_ifgo_to_onnx_iofc() {
        assert_eq!(
            gate_order(&[1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
            [1., 2., 7., 8., 3., 4., 5., 6.]
        );
        let b = binding(&[1, 8], "lstm_bias");
        assert_eq!(
            convert(&b, &[&[1., 2., 3., 4.], &[5., 6., 7., 8.]]).unwrap(),
            [1., 4., 2., 3., 5., 8., 6., 7.]
        );
    }
    #[test]
    fn matrix_and_batch_norm_folding() {
        assert_eq!(
            convert(&binding(&[2, 3], "transpose"), &[&[1., 2., 3., 4., 5., 6.]]).unwrap(),
            [1., 3., 5., 2., 4., 6.]
        );
        let v = convert(
            &binding(&[1, 1, 2], "batch_norm_weight"),
            &[&[2., 4.], &[3.], &[4.]],
        )
        .unwrap();
        assert!((v[0] - 3.).abs() < 1e-5 && (v[1] - 6.).abs() < 1e-5);
        let v = convert(
            &binding(&[1], "batch_norm_bias"),
            &[&[5.], &[3.], &[2.], &[4.]],
        )
        .unwrap();
        assert!((v[0] - 2.).abs() < 1e-5);
    }
    #[test]
    fn templates_have_no_original_weights_and_match_bindings() {
        use onnx_protobuf::{tensor_proto::DataLocation, Message, ModelProto};
        let bindings: Vec<Binding> =
            serde_json::from_str(include_str!("onnx/bindings.json")).unwrap();
        for (name, bytes) in [
            ("encoder", &include_bytes!("onnx/encoder.onnx")[..]),
            ("decoder", &include_bytes!("onnx/decoder.onnx")[..]),
        ] {
            let model = ModelProto::parse_from_bytes(bytes).unwrap();
            assert!(model
                .opset_import
                .iter()
                .all(|op| op.version == 17 && op.domain.is_empty()));
            let graph = model.graph.as_ref().unwrap();
            for t in &graph.initializer {
                assert_eq!(
                    t.data_location.enum_value().unwrap(),
                    DataLocation::EXTERNAL
                );
                assert!(t.raw_data.is_empty() && t.float_data.is_empty());
                let b = bindings
                    .iter()
                    .find(|b| b.graph == name && b.target == t.name)
                    .unwrap();
                assert_eq!(
                    b.shape.iter().map(|&n| n as i64).collect::<Vec<_>>(),
                    t.dims
                );
            }
            assert_eq!(
                graph.initializer.len(),
                bindings.iter().filter(|b| b.graph == name).count()
            );
            for n in &graph.node {
                for a in &n.attribute {
                    if let Some(t) = a.t.as_ref() {
                        assert!(t.raw_data.len() < 1024);
                    }
                }
            }
        }
    }
    #[test]
    fn manifest_covers_each_native_model_tensor() {
        let b: Vec<Binding> = serde_json::from_str(include_str!("onnx/bindings.json")).unwrap();
        let sources: HashSet<_> = b.iter().flat_map(|b| b.sources.iter()).collect();
        assert_eq!(b.len(), 626);
        assert_eq!(sources.len(), 699);
        assert_eq!(b.iter().filter(|b| b.graph == "encoder").count(), 613);
        assert_eq!(b.iter().filter(|b| b.graph == "decoder").count(), 13);
    }
}
