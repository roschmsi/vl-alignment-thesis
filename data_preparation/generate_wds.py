import os
import glob
import logging
import torch
import webdataset as wds
from tqdm import tqdm

# ==============================================================================
# --- ⚙️ CONFIGURATION ---
# ==============================================================================

BASE_OUTPUT_DIR = "/dss/mcmlscratch/07/ga27tus3/webdataset_data"

VISION_REPS_PATH = "/dss/mcmlscratch/07/ga27tus3/tensor_data/image_embedding/dinov2-large/dreamclipcc12m_concat/*.pt"
TEXT_REPS_PATH = "/dss/mcmlscratch/07/ga27tus3/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc12m_raw_caption/*.pt"
EXTRA_TEXT_REPS_PATH = "/dss/mcmlscratch/07/ga27tus3/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc12m_shortSV_captions/*.pt"

OUTPUT_DATASET_NAME = "dreamclipcc12m_individual_simple"
SAMPLES_PER_SHARD = 10000

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_name_from_path(path: str) -> str:
    """Extracts a model name from a given file path string."""
    if not path: return None
    try:
        relevant_part = path.split('/tensor_data/')[1]
        model_path = os.path.dirname(os.path.dirname(relevant_part))
        return os.path.basename(model_path)
    except IndexError:
        return os.path.basename(os.path.dirname(os.path.dirname(path)))

def create_webdataset_from_instances_simple(
    modality_paths: dict,
    base_output_dir: str,
    output_dataset_name: str,
    samples_per_shard: int
):
    """
    Creates a sharded WebDataset by writing individual instances from pre-shuffled chunks.
    """
    active_modalities = {name: path for name, path in modality_paths.items() if path}
    if not active_modalities:
        logging.error("No active modalities provided. Halting.")
        return

    logging.info(f"Active modalities: {list(active_modalities.keys())}")
    logging.info(f"Output dataset name set to: '{output_dataset_name}'")

    # --- 1. Prepare output directory ---
    model_names = [f"{name}-{get_model_name_from_path(path)}" for name, path in active_modalities.items()]
    output_dir_name = f"{'_'.join(model_names)}_{output_dataset_name}"
    output_dir = os.path.join(base_output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"WebDataset shards will be saved to: {output_dir}")

    # --- 2. Get file lists (assuming they are already in a random order) ---
    modality_files = {name: sorted(glob.glob(path)) for name, path in active_modalities.items()}
    num_files = len(next(iter(modality_files.values())))
    if num_files == 0:
        logging.error(f"No source files found. Check paths.")
        return
    logging.info(f"Found {num_files} chunk files to process.")

    # --- 3. Manually control the TarWriter lifecycle for sharding ---
    total_samples_written = 0
    shard_idx = 0
    shard_pattern = os.path.join(output_dir, f"{output_dataset_name}-%06d.tar")
    writer = wds.TarWriter(shard_pattern % shard_idx)
    
    try:
        # Create an iterator for the file paths
        file_path_iterator = list(zip(*modality_files.values()))

        for file_chunk_paths in tqdm(file_path_iterator, desc="Processing chunks"):
            # Load the next chunk
            tensor_chunks = {name: torch.load(path) for name, path in zip(active_modalities.keys(), file_chunk_paths)}
            num_instances_in_chunk = len(next(iter(tensor_chunks.values())))

            # Iterate through the 32 individual instances in the chunk
            for i in range(num_instances_in_chunk):
                if total_samples_written > 0 and total_samples_written % samples_per_shard == 0:
                    writer.close()
                    shard_idx += 1
                    writer = wds.TarWriter(shard_pattern % shard_idx)

                sample_key = f"{total_samples_written:09d}"
                sample = {"__key__": sample_key}
                for name, chunk_tensor in tensor_chunks.items():
                    sample[f"{name}.pth"] = chunk_tensor[i]

                writer.write(sample)
                total_samples_written += 1
    finally:
        if writer:
            writer.close()

    logging.info(f"✅ Preprocessing complete. Wrote {total_samples_written} individual samples into {shard_idx + 1} shards.")

if __name__ == "__main__":
    paths_to_process = {
        'vision': VISION_REPS_PATH,
        'text': TEXT_REPS_PATH,
        'extra_text': EXTRA_TEXT_REPS_PATH
    }

    create_webdataset_from_instances_simple(
        modality_paths=paths_to_process,
        base_output_dir=BASE_OUTPUT_DIR,
        output_dataset_name=OUTPUT_DATASET_NAME,
        samples_per_shard=SAMPLES_PER_SHARD
    )