import os
import argparse
import pandas as pd
from tqdm import tqdm
import webdataset as wds
import gc
import pdb


def generate_lookup_in_chunks(captions_csv_path, key_col, chunksize=100_000):
    print(f"Loading captions in chunks from: {captions_csv_path}")

    url_to_metadata_map = {}
    reader = pd.read_csv(captions_csv_path, dtype=str, chunksize=chunksize, engine="c")

    print(f"Processing CSV in chunks of {chunksize} rows...")

    for chunk in tqdm(reader, desc="Building lookup map"):
        records = chunk.to_dict(orient="records")
        chunk_map = {
            record[key_col]: {k: v for k, v in record.items() if k != key_col}
            for record in records
        }
        url_to_metadata_map.update(chunk_map)

    del reader
    gc.collect()

    print(f"Successfully built the map with {len(url_to_metadata_map):,} entries.")

    return url_to_metadata_map


def main():
    parser = argparse.ArgumentParser(description="Recaption CC3M or CC12M dataset.")
    parser.add_argument(
        "--dataset",
        choices=["cc3m", "cc12m"],
        required=True,
        help="Choose which dataset to process: cc3m or cc12m",
    )
    parser.add_argument(
        "--base_dir",
        default="/dss/mcmlscratch/07/ga27tus3",
        help="Base directory containing datasets",
    )
    args = parser.parse_args()

    if args.dataset == "cc3m":
        src_shard_pattern = f"{args.base_dir}/cc3m/cc3m-train-{{0000..0575}}.tar"
        captions_csv_path = f"{args.base_dir}/cc3m_3long_3short_1raw_captions_url.csv"
        out_dir = f"{args.base_dir}/cc3m_recaptioned"
    elif args.dataset == "cc12m":
        src_shard_pattern = f"{args.base_dir}/cc12m/cc12m-train-{{0000..2175}}.tar"
        captions_csv_path = f"{args.base_dir}/cc12m_3long_3short_1raw_captions_url.csv"
        out_dir = f"{args.base_dir}/cc12m_recaptioned"
    # elif args.dataset == "yfcc15m":
    #     src_shard_pattern = f"{args.base_dir}/cc12m/cc12m-train-{{0000..2175}}.tar"
    #     captions_csv_path = f"{args.base_dir}/cc12m_3long_3short_1raw_captions_url.csv"
    #     out_dir = f"{args.base_dir}/cc12m_recaptioned"

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading new captions from: {captions_csv_path}")

    url_to_metadata_map = generate_lookup_in_chunks(
        captions_csv_path=captions_csv_path, key_col="Image Path", chunksize=500_000
    )

    print(f"Processing shards from: {src_shard_pattern}")
    source_dataset = wds.WebDataset(src_shard_pattern, shardshuffle=False).decode()

    output_path = os.path.join(out_dir, f"{args.dataset}-train-%04d.tar")

    with wds.ShardWriter(output_path, maxcount=10000) as shard_writer:
        updated_count = 0
        total_count = 0

        for sample in tqdm(source_dataset, desc="Rewriting shards"):
            total_count += 1

            metadata = sample.get("json", {})
            lookup_key = metadata.get("url")
            new_captions_data = url_to_metadata_map.get(lookup_key)

            if new_captions_data:
                metadata.update(new_captions_data)
                sample["json"] = metadata
                updated_count += 1
                shard_writer.write(sample)

    print(f"Processed {total_count} total samples.")
    print(f"Found and updated metadata for {updated_count} samples.")


if __name__ == "__main__":
    main()
