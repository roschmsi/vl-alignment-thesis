import torch
import os

output_dir = "/dss/mcmlscratch/07/ga27tus3/mmap_data/dinov2_large_NV_Embed_v2"

metadata = {
    'vision_shape': [2048, 9],
    'text_shape': [4096, 11],
    'vision_dtype': 'float16',
    'text_dtype': 'float16',
    'num_samples': 10012845
}

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'metadata.pt')
torch.save(metadata, output_path)

print(f"Successfully generated metadata file at: {output_path}")