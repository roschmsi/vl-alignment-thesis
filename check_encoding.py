# single_batch_pipeline_check.py
import os
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
import webdataset as wds

# ==== import your project code ====
from model import ImageEmbedding, SentenceEmbedding
import logging
from train.logger import setup_logging
from encode_wds import (
    load_data, pil_collate_fn
)
from encode_wds import encode_text as encode_text_wds
from encode_wds import encode_image as encode_image_wds

setup_logging(log_file=None, level=logging.INFO)

# --------- config (edit these) ----------
DATA_KEY        = "dreamclipcc12m"
SOURCE_CAPTION  = "shortSV_captions"
VISION_MODEL    = "facebook/dinov2-large"
TEXT_MODEL      = "nvidia/NV-Embed-v2"
OUTPUT_BASE     = "/dss/dsshome1/07/ga27tus3/vision-language-alignment/check_emb_output"
START_SHARD     = END_SHARD = 0           # read from the first shard only
sample_start_index = 0
# ----------------------------------------

BASES = {
    "dreamclipcc3m":  "/dss/mcmlscratch/07/ga27tus3/pixparse/cc3m_recaptioned/cc3m-train-",
    "dreamclipcc12m": "/dss/mcmlscratch/07/ga27tus3/pixparse/cc12m_recaptioned/cc12m-train-",
}
base_path = BASES[DATA_KEY]
data_path = f"{base_path}{{{START_SHARD:04d}..{END_SHARD:04d}}}.tar"

args_common = dict(
    data=DATA_KEY,
    vision_model_name=VISION_MODEL,
    text_model_name=TEXT_MODEL,
    resume=False,
    start_index=0,
    end_index=None,
    start_shard_index=START_SHARD,
    end_shard_index=END_SHARD,
    source_caption=SOURCE_CAPTION,
    save_name=None,
    batch_size=1,
    agg_mode="concat",
    throughput=False,
    output_dir=OUTPUT_BASE,
    output_hidden_states=True,
    num_workers=0,
    downsample=False,
)

# --- 1) TEXT: build dataset/loader, keep first batch only, run encode_text ---
args_text = SimpleNamespace(**{**args_common, "domain": "text"})
dataset_text = load_data(data_path, args_text.source_caption, args_text.domain)
loader_text_full = DataLoader(dataset_text, batch_size=5, num_workers=0, shuffle=False)

first_text_batch = next(iter(loader_text_full))  # list[str] with length 1
single_text_loader = [first_text_batch]          # one-batch iterable

print(first_text_batch)

print(">> Running encode_text on a single batch…")
encode_text_wds(args_text, single_text_loader, sample_start_index)

# --- 2) IMAGE: build dataset/loader, keep first batch only, run encode_image ---
args_img = SimpleNamespace(**{**args_common, "domain": "image"})
dataset_img = load_data(data_path, args_img.source_caption, args_img.domain)
loader_img_full = DataLoader(
    dataset_img,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    collate_fn=pil_collate_fn,    # <- same collate you use in the pipeline
)

first_img_batch = next(iter(loader_img_full))    # list[PIL.Image] with length 1
single_img_loader = [first_img_batch]            # one-batch iterable

print(">> Running encode_image on a single batch…")
encode_image_wds(args_img, single_img_loader, sample_start_index)

print("\nDone. Embeddings saved under your standard output tree.")