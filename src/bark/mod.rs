// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
//! Native Underdog Bark ANS/XZ decoding and optional MLX inference.
//! Enable `bark-mlx` and use `Model::from_pretrained("bark-3.0-mlx")`.
//! English only; no Python, converted-weight cache or local server.

#[cfg(feature = "bark-mlx")]
mod engine;
pub mod weights;
#[cfg(feature = "bark-mlx")]
pub(crate) use engine::BarkEngine;

pub const MODEL_NAME: &str = "bark-3.0-mlx";
pub const MODEL_REPO: &str = "ConwayResearch/Underdog-Bark-3.0";
pub const MODEL_REVISION: &str = "85db2ad18ab41e534a222ee253c29191d77ed919";
pub const MODEL_SHA256: &str = "c18de9418d73cbeb80b8c1287b392d480a56e191ef4a81d1baaa1b02e66d92ed";

/// Download immutable model assets, retaining upstream attribution in the cache.
#[cfg(feature = "bark-mlx")]
pub(crate) fn download() -> anyhow::Result<std::path::PathBuf> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        MODEL_REPO.into(),
        hf_hub::RepoType::Model,
        MODEL_REVISION.into(),
    ));
    repo.get("NOTICE")?;
    repo.get("mlx/config.json")?;
    repo.get("model.bark").map_err(Into::into)
}

#[cfg(feature = "bark-mlx")]
pub(crate) fn cached_model() -> Option<std::path::PathBuf> {
    // Some Hub clients omit refs/<revision> for immutable SHAs. Read the pinned snapshot directly so
    // either client can populate the cache without repeated network requests.
    let repo = hf_hub::Repo::model(MODEL_REPO.into());
    let root = hf_hub::Cache::default()
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(MODEL_REVISION);
    ["mlx/config.json", "NOTICE", "model.bark"]
        .iter()
        .all(|f| root.join(f).is_file())
        .then(|| root.join("model.bark"))
}
