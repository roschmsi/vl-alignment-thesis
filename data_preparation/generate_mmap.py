import torch
import numpy as np
import glob
from tqdm import tqdm
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_OUTPUT_DIR = "/dss/mcmlscratch/07/ga27tus3/mmap_data"

VISION_REPS_PATH = None # "/dss/mcmlscratch/07/ga27tus3/tensor_data/image_embedding/dinov2-large/dreamclipcc12m_concat/*.pt"
TEXT_REPS_PATH = "/dss/mcmlscratch/07/ga27tus3/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc12m_raw_caption/*.pt"
EXTRA_TEXT_REPS_PATH = None

def get_model_name_from_path(path: str) -> str:
    if not path: return None
    try:
        relevant_part = path.split('/tensor_data/')[1]
        model_path = os.path.dirname(os.path.dirname(relevant_part))
        return os.path.basename(model_path)
    except IndexError:
        return os.path.basename(os.path.dirname(os.path.dirname(path)))

def get_dataset_name_from_path(path: str) -> str:
    if not path: return None
    return os.path.basename(os.path.dirname(path))

def calculate_total_samples(glob_path: str) -> int:
    """Scans files in a directory to calculate the total number of samples."""
    logging.info(f"Calculating total number of samples from: {glob_path}")
    files = sorted(glob.glob(glob_path))
    if not files:
        raise FileNotFoundError(f"No files found at '{glob_path}'. Cannot determine sample count.")
    
    total_samples = 0
    for file_path in tqdm(files, desc="Counting samples"):
        # Load only the header or a small part if possible, but torch.load is standard
        tensor_chunk = torch.load(file_path, map_location='cpu')
        total_samples += tensor_chunk.shape[0]
    return total_samples

def process_directory(modality_name: str, input_glob_path: str, base_output_dir: str, total_samples: int):
    """
    Processes a single directory of tensor files and saves them to a memory-mapped file.
    """
    logging.info(f"--- Processing modality: {modality_name} ---")
    
    # 1. Prepare Paths
    model_name = get_model_name_from_path(input_glob_path)
    dataset_name = get_dataset_name_from_path(input_glob_path)
    output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    mmap_path = os.path.join(output_dir, f"{dataset_name}.mmap")
    meta_path = os.path.join(output_dir, f"{dataset_name}_meta.pt")
    logging.info(f"Output for {modality_name} will be saved to: {mmap_path}")

    # 2. Get file list and check consistency
    files = sorted(glob.glob(input_glob_path))
    if not files:
        logging.warning(f"No files found for {modality_name} at path {input_glob_path}. Skipping.")
        return None

    # 3. Determine shape and dtype
    first_tensor = torch.load(files[0])
    single_shape = first_tensor.shape[1:]
    dtype = first_tensor.numpy().dtype
    logging.info(f"Detected sample shape: {single_shape} and dtype: {dtype}")

    # 4. Create memory-mapped file
    mmap_file = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=(total_samples, *single_shape))

    # 5. Aggregate data into mmap file
    current_position = 0
    for file_path in tqdm(files, desc=f"Aggregating {modality_name}"):
        tensor_chunk = torch.load(file_path)
        chunk_size = tensor_chunk.shape[0]
        mmap_file[current_position : current_position + chunk_size] = tensor_chunk.numpy()
        current_position += chunk_size
    
    mmap_file.flush()
    
    # 6. Save metadata for this modality
    metadata = {
        'model': model_name,
        'dataset': dataset_name,
        'num_samples': total_samples,
        'shape': single_shape,
        'dtype': str(dtype),
    }

    torch.save(metadata, meta_path)
   
    return


if __name__ == "__main__":
    paths_to_process = {
        'vision': VISION_REPS_PATH,
        'text': TEXT_REPS_PATH,
        'extra_text': EXTRA_TEXT_REPS_PATH
    }

    active_paths = {name: path for name, path in paths_to_process.items() if path is not None}

    if not active_paths:
        logging.error("No input paths are configured. Please set at least one path variable.")
    else:
        # Determine primary path for counting samples
        primary_path_name = next(iter(active_paths))
        primary_path = active_paths[primary_path_name]
        
        num_samples = calculate_total_samples(primary_path)
        logging.info(f"Calculation complete. Total samples to process: {num_samples}")
        
        # Process each directory
        for name, path in active_paths.items():
            process_directory(
                modality_name=name,
                input_glob_path=path,
                base_output_dir=BASE_OUTPUT_DIR,
                total_samples=num_samples
            )
        
        logging.info(f"Preprocessing complete.")