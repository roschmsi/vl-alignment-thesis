import csv
import json
import logging
import os
import time
import warnings
import itertools

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from data.data_config import DATADIR
from model import ImageEmbedding, SentenceEmbedding
from train.logger import setup_logging
import webdataset as wds
import pdb

setup_logging(log_file=None, level=logging.INFO)
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
warnings.filterwarnings("ignore", message="Palette images with Transparency")
warnings.filterwarnings("ignore", message="WebDataset")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Data type (key in DATADIR)")
    parser.add_argument("--vision_model_name", type=str, required=True, help="Model name")
    parser.add_argument("--text_model_name", type=str, required=True, help="Model name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing embeddings")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for data processing")
    parser.add_argument("--end_index", type=int, default=None, help="End index for data processing")
    parser.add_argument("--start_shard_index", type=int, default=0, help="Start shard for data processing")
    parser.add_argument("--end_shard_index", type=int, default=None, help="End shard for data processing")
    parser.add_argument("--domain", type=str, choices=["text", "image"], required=True, help="Domain to encode")
    parser.add_argument(
        "--source_caption",
        type=str,
        choices=[
            "raw_caption", "shortIB_captions", "longIB_captions",
            "shortSV_captions", "longSV_captions", "shortLLA_captions",
            "longLLA_captions", "caption", "txt"
        ],
        default="raw_caption",
        help="Source caption key inside the .json file in the webdataset",
    )
    parser.add_argument("--save_name", type=str, default=None, help="Save name suffix for output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Processing batch size")
    parser.add_argument("--agg_mode", type=str, default="concat", help="Aggregation mode for vision models")
    parser.add_argument("--throughput", action="store_true", help="Measure throughput without saving")
    parser.add_argument("--output_dir", type=str, default="./data", help="Base directory to store embeddings")
    parser.add_argument("--output_hidden_states", action="store_true", help="Output the hidden states of a model")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--downsample", action="store_true", help="Only store hidden representations after every n-th layer")
    return parser.parse_args()


def process_batch_from_loader(data_loader, model_func, start_index, batch_size, output_dir, resume, downsample=False, throughput=False):
    idx = start_index // batch_size
    total_time = 0
    total_samples = 0
    for batch_data in tqdm(data_loader, desc="Encoding Batches"):
        output_path = os.path.join(output_dir, f"{idx}.pt")
        if resume and os.path.exists(output_path):
            idx += 1
            continue
        start_time = time.time()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                batch_embeddings = model_func(batch_data).cpu()
        end_time = time.time()
        batch_time = end_time - start_time
        num_samples_in_batch = len(batch_data) if isinstance(batch_data, list) else batch_data.size(0)
        total_time += batch_time
        total_samples += num_samples_in_batch
        current_throughput = num_samples_in_batch / batch_time
        avg_throughput = total_samples / total_time

        if downsample:
            L = batch_embeddings.shape[-1]
            start = (L - 1) % 3
            batch_embeddings = batch_embeddings[:, :, start::3]

        if throughput:
            logging.info(f"Batch {idx} throughput: {current_throughput:.2f} samples/sec, Avg: {avg_throughput:.2f} samples/sec")
        else:
            batch_embeddings = batch_embeddings.half()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
        idx += 1
    if total_samples > 0 and total_time > 0:
        final_throughput = total_samples / total_time
        logging.info(f"Final average throughput: {final_throughput:.2f} samples/sec")
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(f"Total samples processed: {total_samples}")


@torch.no_grad()
def encode_text(args, data_loader, start_index):
    model_name = args.text_model_name.split("/")[-1]
    save_suffix = args.save_name if args.save_name else ""
    output_dir = os.path.join(
        f"{args.output_dir}/tensor_data/text_embedding",
        model_name,
        f"{args.data}_{args.source_caption}_{save_suffix}".strip("_"),
    )
    print(f"Output directory: {output_dir}")
    model = SentenceEmbedding(args.text_model_name, output_hidden_states=args.output_hidden_states)
    model = model.half().to("cuda")
    model.eval()
    def encode_function(batch_sentences):
        return model.get_sentence_embeddings(list(batch_sentences))
    process_batch_from_loader(data_loader, encode_function, start_index, args.batch_size, output_dir, args.resume, args.downsample, args.throughput)


@torch.no_grad()
def encode_image(args, data_loader, start_index):
    model_name = args.vision_model_name.split("/")[-1]
    output_dir = os.path.join(
        f"{args.output_dir}/tensor_data/image_embedding",
        model_name,
        f"{args.data}_{args.agg_mode}",
    )
    print(f"Output directory: {output_dir}")

    # Instantiate the model from your custom class
    model = ImageEmbedding(
        args.vision_model_name, 
        agg_mode=args.agg_mode, 
        output_hidden_states=args.output_hidden_states
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
            return sample['jpg']
        dataset = dataset.map(image_extractor, handler=wds.warn_and_continue)
    
    elif domain == "text":
        def text_extractor(sample):
            return sample['json'][source_caption]
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
    
    # data_path = f"/dss/mcmlscratch/07/ga27tus3/pixparse/cc3m_recaptioned/cc3m-recaptioned-{{000000..000001}}.tar"
    if args.data == "dreamclipcc3m":
        base_path = "/dss/mcmlscratch/07/ga27tus3/pixparse/cc3m_recaptioned/cc3m-train-"
        max_shard_index = 282
    elif args.data == "dreamclipcc12m":
        base_path = "/dss/mcmlscratch/07/ga27tus3/pixparse/cc12m_recaptioned/cc12m-train-"
        max_shard_index = 1001
    else:
        raise ValueError(f"Unknown data: {args.data}.")

    start_index = args.start_shard_index if args.start_shard_index is not None else 0
    end_index = args.end_shard_index if args.end_shard_index is not None else max_shard_index

    data_path = f"{base_path}{{{start_index:04d}..{end_index:04d}}}.tar"
    
    logging.info(f"Processing shards from index {start_index} to {end_index}.")
    logging.info(f"Constructed WebDataset path: {data_path}")

    dataset = load_data(data_path, args.source_caption, args.domain)

    samples_per_shard = 10000
    sample_start_index = start_index * samples_per_shard

    if args.domain == 'text':
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        logging.info(f"Encoding text data '{args.data}' ({args.source_caption}) with model {args.text_model_name}...")
        encode_text(args, data_loader, sample_start_index)

    elif args.domain == 'image':
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=pil_collate_fn
        )
        logging.info(f"Encoding image data '{args.data}' with model {args.vision_model_name}...")
        encode_image(args, data_loader, sample_start_index)


if __name__ == "__main__":
    main()