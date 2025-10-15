from huggingface_hub import snapshot_download
import os

repo_id = "pixparse/cc3m-wds"
local_dir = "/dss/mcmlscratch/07/ga27tus3/pixparse/cc3m"

# repo_id = "Kaichengalex/YFCC15M"
# local_dir = "/dss/mcmlscratch/07/ga27tus3/yfcc15m"

print(f"Downloading shards from {repo_id} to {local_dir}...")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    # allow_patterns=["*.tar"]
)

print("Download complete.")

print("Listing downloaded shards:")

if os.path.exists(local_dir):
    downloaded_shards = os.listdir(local_dir)
    for shard in downloaded_shards[:10]:
        print(f"- {shard}")
    print(f"... and {len(downloaded_shards) - 10} more files.")
else:
    print("Could not find the 'data' directory. Check the download.")