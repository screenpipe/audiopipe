// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
//! Strip learned weights from pinned Parakeet architecture graphs and describe
//! how native Bark tensors populate them. No Python or ONNX exporter required.
use anyhow::{ensure, Context, Result};
use onnx_protobuf::{tensor_proto::DataLocation, Message, ModelProto, StringStringEntryProto};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

fn source(name: &str) -> String {
    let mut name = name.to_string();
    for (nemo, bark) in [
        ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
        ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
        (".conv.batch_norm.", ".conv.norm."),
        (".linear_q.", ".q_proj."),
        (".linear_k.", ".k_proj."),
        (".linear_v.", ".v_proj."),
        (".linear_out.", ".o_proj."),
        (".linear_pos.", ".relative_k_proj."),
        (".pos_bias_u", ".bias_u"),
        (".pos_bias_v", ".bias_v"),
        ("decoder.prediction.embed.", "decoder.embedding."),
        ("joint.enc.", "encoder_projector."),
        ("joint.pred.", "decoder.decoder_projector."),
        ("joint.joint_net.2.", "joint.head."),
    ] {
        name = name.replace(nemo, bark);
    }
    name
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "prepare_bark_onnx ENCODER.onnx DECODER.onnx OUT_DIR"
    );
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out)?;
    let mut bindings = Vec::new();
    for (kind, path) in [("encoder", &a[1]), ("decoder", &a[2])] {
        let mut model = ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
        ensure!(
            model
                .opset_import
                .iter()
                .all(|x| x.domain.is_empty() && x.version == 17),
            "unexpected opset"
        );
        let graph = model.graph.as_mut().context("missing graph")?;
        let mut rules: BTreeMap<String, (Vec<String>, String)> = BTreeMap::new();
        let mut layer = 0;
        for node in &graph.node {
            if node.op_type == "MatMul"
                && node.input.get(1).is_some_and(|s| s.starts_with("onnx::"))
            {
                let stem = node
                    .name
                    .trim_start_matches('/')
                    .trim_end_matches("/MatMul")
                    .replace('/', ".")
                    .replace("joint_net.joint_net.", "joint_net.");
                let name = if kind == "encoder" {
                    format!("encoder.{stem}.weight")
                } else {
                    format!("{stem}.weight")
                };
                rules.insert(
                    node.input[1].clone(),
                    (vec![source(&name)], "transpose".into()),
                );
            } else if node.op_type == "Conv"
                && node.input.get(1).is_some_and(|s| s.starts_with("onnx::"))
            {
                ensure!(
                    node.name.ends_with("/conv/depthwise_conv/Conv"),
                    "unexpected folded Conv"
                );
                let stem = format!(
                    "encoder.{}",
                    node.name
                        .trim_start_matches('/')
                        .trim_end_matches("/depthwise_conv/Conv")
                        .replace('/', ".")
                );
                rules.insert(
                    node.input[1].clone(),
                    (
                        vec![
                            format!("{stem}.depthwise_conv.weight"),
                            format!("{stem}.norm.weight"),
                            format!("{stem}.norm.running_var"),
                        ],
                        "batch_norm_weight".into(),
                    ),
                );
                rules.insert(
                    node.input[2].clone(),
                    (
                        vec![
                            format!("{stem}.norm.bias"),
                            format!("{stem}.norm.weight"),
                            format!("{stem}.norm.running_mean"),
                            format!("{stem}.norm.running_var"),
                        ],
                        "batch_norm_bias".into(),
                    ),
                );
            } else if node.op_type == "LSTM" {
                rules.insert(
                    node.input[1].clone(),
                    (
                        vec![format!("decoder.lstm.weight_ih_l{layer}")],
                        "lstm".into(),
                    ),
                );
                rules.insert(
                    node.input[2].clone(),
                    (
                        vec![format!("decoder.lstm.weight_hh_l{layer}")],
                        "lstm".into(),
                    ),
                );
                rules.insert(
                    node.input[3].clone(),
                    (
                        vec![
                            format!("decoder.lstm.bias_ih_l{layer}"),
                            format!("decoder.lstm.bias_hh_l{layer}"),
                        ],
                        "lstm_bias".into(),
                    ),
                );
                layer += 1;
            }
        }
        // Generate sinusoidal positions at runtime instead of embedding 41 MB.
        let mut positions = Vec::new();
        graph.node.retain(|node| {
            if node.op_type == "Constant" {
                for a in &node.attribute {
                    if let Some(t) = a.t.as_ref() {
                        if t.raw_data.len() > 1_000_000 {
                            assert_eq!(t.dims, [1, 9999, 1024]);
                            let mut t = t.clone();
                            t.name = node.output[0].clone();
                            rules.insert(t.name.clone(), (vec![], "position".into()));
                            positions.push(t);
                            return false;
                        }
                    }
                }
            }
            true
        });
        graph.initializer.extend(positions);
        ensure!(
            kind != "encoder" || graph.initializer.len() == 613,
            "unexpected encoder initializer count"
        );
        ensure!(
            kind != "decoder" || graph.initializer.len() == 13,
            "unexpected decoder initializer count"
        );
        for t in &mut graph.initializer {
            ensure!(t.data_type == 1, "non-f32 initializer {}", t.name);
            let (sources, transform) = if let Some(r) = rules.remove(&t.name) {
                r
            } else {
                ensure!(
                    !t.name.starts_with("onnx::"),
                    "unmapped initializer {}",
                    t.name
                );
                let n = if kind == "encoder" {
                    format!("encoder.{}", t.name)
                } else {
                    t.name.clone()
                };
                (vec![source(&n)], "identity".to_string())
            };
            bindings.push(json!({"graph":kind,"target":t.name,"shape":t.dims,"sources":sources,"transform":transform}));
            t.raw_data.clear();
            t.float_data.clear();
            t.double_data.clear();
            t.int32_data.clear();
            t.int64_data.clear();
            t.uint64_data.clear();
            t.string_data.clear();
            t.data_location = DataLocation::EXTERNAL.into();
            t.external_data = vec![
                StringStringEntryProto {
                    key: "location".into(),
                    value: "bark.native".into(),
                    ..Default::default()
                },
                StringStringEntryProto {
                    key: "length".into(),
                    value: (t.dims.iter().product::<i64>() * 4).to_string(),
                    ..Default::default()
                },
            ];
        }
        ensure!(rules.is_empty(), "unused conversion rules");
        model.producer_name = "audiopipe native Bark architecture template".into();
        std::fs::write(out.join(format!("{kind}.onnx")), model.write_to_bytes()?)?;
    }
    let unique: BTreeSet<_> = bindings
        .iter()
        .flat_map(|v| {
            v["sources"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap())
        })
        .collect();
    eprintln!(
        "{} initializers, {} unique Bark tensors",
        bindings.len(),
        unique.len()
    );
    std::fs::write(
        out.join("bindings.json"),
        serde_json::to_vec_pretty(&Value::Array(bindings))?,
    )?;
    Ok(())
}
