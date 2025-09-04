import pandas as pd
import webdataset as wds
import glob
import json

# --- Configure paths from your script ---
base_dir = "/dss/mcmlscratch/07/ga27tus3"
src_shard_pattern = f"{base_dir}/pixparse/cc3m/cc3m-train-*.tar"
captions_csv_path = f"{base_dir}/cc3m_3long_3short_1raw_captions_url.csv"

# --- 1. Inspect the CSV file ---
print("--- 1. Inspecting the CSV file ---")
try:
    df = pd.read_csv(captions_csv_path)
    print("Column names in the CSV:")
    print(df.columns.tolist())
    print("\nFirst 3 rows of the CSV:")
    print(df.head(3).to_markdown())
except Exception as e:
    print(f"Could not read CSV: {e}")

# --- 2. Inspect a sample from the WebDataset ---
print("\n\n--- 2. Inspecting a sample from the WebDataset shard ---")
try:
    # Find the first shard file
    first_shard = sorted(glob.glob(src_shard_pattern))[0]
    print(f"Reading from shard: {first_shard}")

    # Create a WebDataset object and decode the contents
    dataset = wds.WebDataset(first_shard).decode("rgb", "json")

    # Get the very first sample from the shard
    sample = next(iter(dataset))
    
    print("\nKeys available in a WebDataset sample:")
    print(list(sample.keys()))

    print("\nContents of the sample's .json file:")
    # Pretty-print the JSON content for readability
    print(json.dumps(sample.get('json', {}), indent=2))

except Exception as e:
    print(f"Could not read WebDataset shard: {e}")