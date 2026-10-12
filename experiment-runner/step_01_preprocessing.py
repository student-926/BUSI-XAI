import pandas as pd
from busi_xai.preprocessing import preprocess_dataset
from busi_xai.config import METRICS_DIR


print("=" * 60, flush=True)
print("Start preprocessing data", flush=True)
print("=" * 60, flush=True)
results = preprocess_dataset()
df = pd.DataFrame(results)
print("=" * 60, flush=True)
print("Finished preprocessing data", flush=True)
print("=" * 60, flush=True)

METRICS_DIR.mkdir(parents=True, exist_ok=True)
manifest_path = METRICS_DIR / "preprocessing_manifest.csv"
df.to_csv(manifest_path, index=False)
print(f"Saved: {manifest_path}")