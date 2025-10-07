import torch
import numpy as np
import glob
from tqdm import tqdm
import os

VISION_REPS_PATH = "/dss/mcmlscratch/07/ga27tus3/tensor_data/image_embedding/dinov2-large/dreamclipcc12m_concat/*.pt"
TEXT_REPS_PATH = "/dss/mcmlscratch/07/ga27tus3/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc12m_longSV_captions/*.pt"
BASE_OUTPUT_DIR = "/dss/mcmlscratch/07/ga27tus3/mmap_data"

def get_model_name_from_path(path: str) -> str:
    relevant_part = path.split('/tensor_data/')[1]
    model_path = os.path.dirname(os.path.dirname(relevant_part))
    return os.path.basename(model_path)

def get_dataset_name_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))

vision_model_name = get_model_name_from_path(VISION_REPS_PATH)
text_model_name = get_model_name_from_path(TEXT_REPS_PATH)

vision_dataset_name = get_dataset_name_from_path(VISION_REPS_PATH)
text_dataset_name = get_dataset_name_from_path(TEXT_REPS_PATH)

os.makedirs(os.path.join(BASE_OUTPUT_DIR, vision_model_name), exist_ok=True)
os.makedirs(os.path.join(BASE_OUTPUT_DIR, text_model_name), exist_ok=True)

vision_mmap_path = os.path.join(BASE_OUTPUT_DIR, vision_model_name, f"{vision_dataset_name}.mmap")
text_mmap_path = os.path.join(BASE_OUTPUT_DIR, text_model_name, f"{text_dataset_name}.mmap")

print(f"Vision mmap will be saved to: {vision_mmap_path}")
print(f"Text mmap will be saved to: {text_mmap_path}")

print("Scanning for representation files...")
vision_files = sorted(glob.glob(VISION_REPS_PATH))
text_files = sorted(glob.glob(TEXT_REPS_PATH))

if not vision_files:
    raise FileNotFoundError("No representation files found. Check your paths.")
assert len(vision_files) == len(text_files), "Vision and text file counts do not match!"

total_samples = 0
print("Calculating total number of samples...")
for file_path in tqdm(vision_files):
    tensor_chunk = torch.load(file_path)
    total_samples += tensor_chunk.shape[0]

print(f"Calculation complete. Exact total samples: {total_samples}")

# --- Determine tensor shapes and dtypes from the first file ---
first_vision_tensor = torch.load(vision_files[0])
first_text_tensor = torch.load(text_files[0])
vision_single_shape = first_vision_tensor.shape[1:]
text_single_shape = first_text_tensor.shape[1:]
vision_dtype = first_vision_tensor.numpy().dtype
text_dtype = first_text_tensor.numpy().dtype

print(f"Single vision sample shape: {vision_single_shape}")
print(f"Single text sample shape: {text_single_shape}")

# --- Create memory-mapped files with the exact shape ---
vision_mmap = np.memmap(vision_mmap_path, dtype=vision_dtype, mode='w+', shape=(total_samples, *vision_single_shape))
text_mmap = np.memmap(text_mmap_path, dtype=text_dtype, mode='w+', shape=(total_samples, *text_single_shape))

print("Aggregating vision representations...")
current_position = 0
for file_path in tqdm(vision_files):
    tensor_chunk = torch.load(file_path)
    chunk_size = tensor_chunk.shape[0]
    vision_mmap[current_position : current_position + chunk_size] = tensor_chunk.numpy()
    current_position += chunk_size

print("Aggregating text representations...")
current_position = 0
for file_path in tqdm(text_files):
    tensor_chunk = torch.load(file_path)
    chunk_size = tensor_chunk.shape[0]
    text_mmap[current_position : current_position + chunk_size] = tensor_chunk.numpy()
    current_position += chunk_size

vision_mmap.flush()
text_mmap.flush()

print(f"Preprocessing complete! Data for {total_samples} samples saved to {BASE_OUTPUT_DIR}.")

metadata = {
    'vision_shape': vision_single_shape,
    'vision_dtype': str(vision_dtype),
    'text_shape': text_single_shape,
    'text_dtype': str(text_dtype),
    'num_samples': total_samples
}
torch.save(metadata, os.path.join(BASE_OUTPUT_DIR, 'metadata.pt'))