import logging
import os
import time
import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ImageEmbedding, SentenceEmbedding
from train.logger import setup_logging
import webdataset as wds
import h5py

setup_logging(log_file=None, level=logging.INFO)
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
warnings.filterwarnings("ignore", message="Palette images with Transparency")
warnings.filterwarnings("ignore", message="WebDataset")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Data type")
    parser.add_argument(
        "--vision_model_name", type=str, required=True, help="Model name"
    )
    parser.add_argument("--text_model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing embeddings"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start index for data processing"
    )
    parser.add_argument(
        "--end_index", type=int, default=None, help="End index for data processing"
    )
    parser.add_argument(
        "--start_shard_index",
        type=int,
        default=0,
        help="Start shard for data processing",
    )
    parser.add_argument(
        "--end_shard_index",
        type=int,
        default=None,
        help="End shard for data processing",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["text", "image"],
        required=True,
        help="Domain to encode",
    )
    parser.add_argument(
        "--source_caption",
        type=str,
        choices=[
            "raw_caption",
            "shortIB_captions",
            "longIB_captions",
            "shortSV_captions",
            "longSV_captions",
            "shortLLA_captions",
            "longLLA_captions",
            "caption",
            "txt",
        ],
        default="raw_caption",
        help="Source caption key inside the .json file in the webdataset",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Save name suffix for output directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Processing batch size"
    )
    parser.add_argument(
        "--agg_mode",
        type=str,
        default="concat",
        help="Aggregation mode for vision models",
    )
    parser.add_argument(
        "--throughput", action="store_true", help="Measure throughput without saving"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data",
        help="Directory to downloaded shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Base directory to store embeddings",
    )
    parser.add_argument(
        "--output_hidden_states",
        action="store_true",
        help="Output the hidden states of a model",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Only store hidden representations after every n-th layer",
    )
    return parser.parse_args()


def process_batch_from_loader(
    data_loader,
    model_func,
    start_index,
    batch_size,
    output_dir,
    resume,
    downsample=False,
    throughput=False,
    hdf5_path=None,
):
    """
    Stream embeddings into a single HDF5 file at hdf5_path.
    Creates dataset lazily on first batch (so we know feature shape).
    Supports resume by appending to existing dataset.
    """
    assert hdf5_path is not None, "hdf5_path must be provided"
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    # State
    total_time = 0.0
    total_samples = 0
    global_write_index = 0  # how many rows already persisted
    h5f = None
    dset = None

    # If resuming, open + read current shape
    if resume and os.path.exists(hdf5_path):
        h5f = h5py.File(hdf5_path, "r+")
        if "embeddings" in h5f:
            dset = h5f["embeddings"]
            global_write_index = dset.shape[0]
            logging.info(f"Resuming HDF5 at row {global_write_index}.")
        else:
            # File exists but no dataset; treat as fresh
            h5f.close()
            h5f = None

    # For consuming batches we’ve already written (if resume),
    # we’ll simply keep counting and skip writing until we've passed
    # global_write_index samples.
    skipped_so_far = 0

    # Iterate
    for batch_data in tqdm(data_loader, desc="Encoding Batches"):
        # Forward pass
        start_t = time.time()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                batch_embeddings = model_func(batch_data)  # torch.Tensor
        end_t = time.time()

        # Bookkeeping
        num_in_batch = (
            len(batch_data) if isinstance(batch_data, list) else batch_data.size(0)
        )
        batch_time = end_t - start_t
        total_time += batch_time
        total_samples += num_in_batch

        # Optional downsample along last dim (your current behavior)
        if downsample:
            L = batch_embeddings.shape[-1]
            start = (L - 1) % 3
            batch_embeddings = (
                batch_embeddings[:, :, start::3]
                if batch_embeddings.ndim >= 2
                else batch_embeddings
            )

        # Throughput-only mode: skip saving entirely
        if throughput:
            current_tp = num_in_batch / max(batch_time, 1e-8)
            avg_tp = total_samples / max(total_time, 1e-8)
            logging.info(
                f"Batch throughput: {current_tp:.2f} samples/sec, Avg: {avg_tp:.2f} samples/sec"
            )
            continue

        # Convert dtype to float16 (to match original .half() saving)
        emb_np = batch_embeddings.detach().to("cpu").numpy().astype(np.float16)

        # Lazy create file + dataset once we know feature shape
        if h5f is None:
            h5f = h5py.File(hdf5_path, "w")
            # Feature shape per sample (exclude batch dimension)
            feat_shape = emb_np.shape[1:]
            # Pick a reasonable chunk size (≤ 4096 rows, ≥ 1)
            chunk_rows = min(max(1, len(emb_np)), 4096)
            dset = h5f.create_dataset(
                "embeddings",
                shape=(0, *feat_shape),
                maxshape=(None, *feat_shape),
                chunks=(chunk_rows, *feat_shape),
                dtype=np.float16,
                compression="gzip",
                compression_opts=1,
            )
            # Metadata (optional but handy)
            dset.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            dset.attrs["start_index"] = start_index
            dset.attrs["batch_size"] = batch_size
            logging.info(
                f"Created HDF5 dataset with per-sample shape {feat_shape} at {hdf5_path}."
            )

        # If resuming and we have already-written rows, skip writing until we catch up
        if skipped_so_far < global_write_index:
            # How many rows from this batch should be skipped?
            remaining_to_skip = global_write_index - skipped_so_far
            if remaining_to_skip >= num_in_batch:
                skipped_so_far += num_in_batch
                # still not caught up; skip entire batch
                continue
            else:
                # Partially skip; write the tail of the batch
                emb_np = emb_np[remaining_to_skip:]
                skipped_so_far += remaining_to_skip
                num_in_batch = emb_np.shape[0]
                if num_in_batch == 0:
                    continue  # nothing to write this iteration

        # Resize and append
        old_n = dset.shape[0]
        new_n = old_n + num_in_batch
        dset.resize((new_n, *dset.shape[1:]))
        dset[old_n:new_n] = emb_np

        # Optional: flush periodically for safety
        if new_n % (batch_size * 10) == 0:
            h5f.flush()

        # Log throughput
        current_tp = (emb_np.shape[0]) / max(batch_time, 1e-8)
        avg_tp = total_samples / max(total_time, 1e-8)
        logging.info(
            f"Saved rows [{old_n}:{new_n}) — batch throughput: {current_tp:.2f} samples/sec, Avg: {avg_tp:.2f} samples/sec"
        )

    if (not throughput) and (h5f is not None):
        h5f.flush()
        h5f.close()

    if total_samples > 0 and total_time > 0:
        final_tp = total_samples / total_time
        logging.info(f"Final average throughput: {final_tp:.2f} samples/sec")
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(f"Total samples processed: {total_samples}")


# def process_batch_from_loader(
#     data_loader,
#     model_func,
#     start_index,
#     batch_size,
#     output_dir,
#     resume,
#     downsample=False,
#     throughput=False,
# ):
#     idx = start_index // batch_size
#     total_time = 0
#     total_samples = 0
#     for batch_data in tqdm(data_loader, desc="Encoding Batches"):
#         output_path = os.path.join(output_dir, f"{idx}.pt")
#         if resume and os.path.exists(output_path):
#             idx += 1
#             continue
#         start_time = time.time()
#         with torch.cuda.amp.autocast():
#             with torch.no_grad():
#                 batch_embeddings = model_func(batch_data).cpu()
#         end_time = time.time()
#         batch_time = end_time - start_time
#         num_samples_in_batch = (
#             len(batch_data) if isinstance(batch_data, list) else batch_data.size(0)
#         )
#         total_time += batch_time
#         total_samples += num_samples_in_batch
#         current_throughput = num_samples_in_batch / batch_time
#         avg_throughput = total_samples / total_time

#         # TODO check that there is only a single downsampling step
#         if downsample:
#             L = batch_embeddings.shape[-1]
#             start = (L - 1) % 3
#             batch_embeddings = batch_embeddings[:, :, start::3]

#         if throughput:
#             logging.info(
#                 f"Batch {idx} throughput: {current_throughput:.2f} samples/sec, Avg: {avg_throughput:.2f} samples/sec"
#             )
#         else:
#             batch_embeddings = batch_embeddings.half()
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             torch.save(batch_embeddings, output_path)
#         idx += 1
#     if total_samples > 0 and total_time > 0:
#         final_throughput = total_samples / total_time
#         logging.info(f"Final average throughput: {final_throughput:.2f} samples/sec")
#         logging.info(f"Total processing time: {total_time:.2f} seconds")
#         logging.info(f"Total samples processed: {total_samples}")


@torch.no_grad()
def encode_text(args, data_loader, start_index):
    model_name = args.text_model_name.split("/")[-1]
    output_dir = os.path.join(
        f"{args.output_dir}/tensor_data/text_embedding",
        model_name,
    )
    print(f"Output directory: {output_dir}")

    hdf5_path = os.path.join(output_dir, f"{args.data}_{args.source_caption}.h5")

    model = SentenceEmbedding(
        args.text_model_name, output_hidden_states=args.output_hidden_states
    )
    model = model.half().to("cuda")
    model.eval()

    def encode_function(batch_sentences):
        return model.get_sentence_embeddings(list(batch_sentences))

    process_batch_from_loader(
        data_loader,
        encode_function,
        start_index,
        args.batch_size,
        output_dir,
        args.resume,
        args.downsample,
        args.throughput,
        hdf5_path=hdf5_path,
    )


@torch.no_grad()
def encode_image(args, data_loader, start_index):
    model_name = args.vision_model_name.split("/")[-1]
    output_dir = os.path.join(
        f"{args.output_dir}/tensor_data/image_embedding",
        model_name,
    )
    print(f"Output directory: {output_dir}")

    hdf5_path = os.path.join(output_dir, f"{args.data}_{args.agg_mode}.h5")

    # Instantiate the model from your custom class
    model = ImageEmbedding(
        args.vision_model_name,
        agg_mode=args.agg_mode,
        output_hidden_states=args.output_hidden_states,
    )
    model = model.to("cuda")
    model.eval()

    def encode_function(batch_of_pil_images):
        return model.get_visual_embeddings_from_pil_list(batch_of_pil_images)

    process_batch_from_loader(
        data_loader,
        encode_function,
        start_index,
        args.batch_size,
        output_dir,
        args.resume,
        args.downsample,
        args.throughput,
        hdf5_path=hdf5_path,
    )


def load_data(data_path, source_caption, domain):
    """
    Loads and prepares data from a webdataset source.
    This version is corrected to work with the enriched CC3M dataset structure.
    """
    logging.info(f"Setting up webdataset from path: {data_path}")

    dataset = wds.WebDataset(data_path, shardshuffle=False, resampled=False)
    dataset = dataset.decode("pil")

    if domain == "image":

        def image_extractor(sample):
            return sample["jpg"]

        dataset = dataset.map(image_extractor, handler=wds.warn_and_continue)

    elif domain == "text":

        def text_extractor(sample):
            return sample["json"][source_caption]

        dataset = dataset.map(text_extractor, handler=wds.warn_and_continue)

    return dataset


def pil_collate_fn(batch):
    """
    Collate function that returns a list of PIL Images.
    This is used when the dataset yields individual PIL Images.
    """
    return list(batch)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.data == "cc3m":
        base_path = f"{args.input_dir}/cc3m-train-"
        max_shard_index = 282
    elif args.data == "cc12m":
        base_path = f"{args.input_dir}/cc12m-train-"
        max_shard_index = 1001
    else:
        raise ValueError(f"Unknown data: {args.data}.")

    start_index = args.start_shard_index if args.start_shard_index is not None else 0
    end_index = (
        args.end_shard_index if args.end_shard_index is not None else max_shard_index
    )

    data_path = f"{base_path}{{{start_index:04d}..{end_index:04d}}}.tar"

    logging.info(f"Processing shards from index {start_index} to {end_index}.")
    logging.info(f"Constructed WebDataset path: {data_path}")

    dataset = load_data(data_path, args.source_caption, args.domain)

    samples_per_shard = 10000
    sample_start_index = start_index * samples_per_shard

    if args.domain == "text":
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        logging.info(
            f"Encoding text data '{args.data}' ({args.source_caption}) with model {args.text_model_name}..."
        )
        encode_text(args, data_loader, sample_start_index)

    elif args.domain == "image":
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=pil_collate_fn,
        )
        logging.info(
            f"Encoding image data '{args.data}' with model {args.vision_model_name}..."
        )
        encode_image(args, data_loader, sample_start_index)


if __name__ == "__main__":
    main()
