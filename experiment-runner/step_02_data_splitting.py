import json
from busi_xai.splitting import create_splits, SPLIT_CONFIG
from busi_xai.config import METRICS_DIR, SPLITS_DIR

print("=" * 60, flush=True)
print("Start splitting data", flush=True)
print("=" * 60, flush=True)
manifest_path = METRICS_DIR / "preprocessing_manifest.csv"
train_df, validation_df, test_df = create_splits(manifest_path)
print("=" * 60, flush=True)
print("Finish splitting data", flush=True)
print("=" * 60, flush=True)

config_path = SPLITS_DIR / "split_config.json"
with open(config_path, "w") as f:
    json.dump(SPLIT_CONFIG, f, indent=2)

print(f"Saved: {config_path}")