// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
fn main() -> Result<(), Box<dyn std::error::Error>> {
    audiopipe::Model::download_pretrained("bark-3.0")?;
    let repo = hf_hub::Repo::model(audiopipe::bark::MODEL_REPO.into());
    let path = hf_hub::Cache::default()
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(audiopipe::bark::MODEL_REVISION)
        .join("model.bark");
    println!("{}", serde_json::json!({"model_path":path}));
    Ok(())
}
