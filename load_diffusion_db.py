import pandas as pd
import os

parquet_url = (
    "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
)

print(f"Downloading and loading metadata from {parquet_url}...")

df = pd.read_parquet(parquet_url)
unique_prompts_df = df[["prompt"]].drop_duplicates()

print(f"Loaded {len(df)} rows.")
print(f"Found {len(unique_prompts_df)} unique prompts.")

output_dir = "/lustre/groups/eml/projects/sroschmann/diffusion_db"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "unique_prompts.csv")

unique_prompts_df.to_csv(output_file, index=False, escapechar="\\")

print(f"Saved unique prompts to: {output_file}")
