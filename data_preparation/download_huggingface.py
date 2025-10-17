import argparse
import os
from huggingface_hub import snapshot_download

DATASET_TO_REPO = {
    "cc3m": "pixparse/cc3m-wds",
    "cc12m": "pixparse/cc12m-wds",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download webdataset shards from Hugging Face hub."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_TO_REPO.keys(),
        required=True,
        help="Which dataset to download: cc3m or cc12m.",
    )
    parser.add_argument(
        "--output-dir",
        default="/lustre/groups/eml/projects/",
        type=str,
        help="Directory to store the downloaded dataset. ",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_id = DATASET_TO_REPO[args.dataset]
    local_dir = os.path.join(args.output_dir, args.dataset)

    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading shards from {repo_id} to {local_dir}...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        # allow_patterns=["*.tar"]
    )

    print("Download complete.")
    print("Listing downloaded shards:")

    if os.path.isdir(local_dir):
        downloaded_shards = sorted(os.listdir(local_dir))
        n = len(downloaded_shards)
        if n == 0:
            print("- (no files found)")
            return
        preview_count = min(10, n)
        for shard in downloaded_shards[:preview_count]:
            print(f"- {shard}")
        if n > preview_count:
            print(f"... and {n - preview_count} more files.")
        else:
            print(f"(total files: {n})")
    else:
        print(
            f"Could not find the directory '{local_dir}'. Check the download path and permissions."
        )


if __name__ == "__main__":
    main()
