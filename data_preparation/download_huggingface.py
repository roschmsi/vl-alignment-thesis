from huggingface_hub import snapshot_download
import os

# Define the repository ID and the local directory to save files
repo_id = "pixparse/cc3m-wds"
local_dir = "/dss/mcmlscratch/07/ga27tus3/pixparse/cc3m"

print(f"Downloading shards from {repo_id} to {local_dir}...")

# Use snapshot_download to get the files
# The `allow_patterns` argument is used to only download the .tar files from the 'data' directory.
# This is highly recommended to avoid downloading other repository files you may not need.
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    # allow_patterns=["*.tar"]
)

print("\nDownload complete.")
print("Listing downloaded shards:")

# List the downloaded files to verify
if os.path.exists(local_dir):
    downloaded_shards = os.listdir(local_dir)
    for shard in downloaded_shards[:10]: # Print first 10 for brevity
        print(f"- {shard}")
    print(f"... and {len(downloaded_shards) - 10} more files.")
else:
    print("Could not find the 'data' directory. Check the download.")