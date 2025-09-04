import webdataset as wds
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
import os
import pdb

# 1. SET THE PATH to the single .tar file you want to inspect.
# This should point to a file in your OUTPUT directory.
tar_path = "/dss/mcmlscratch/07/ga27tus3/pixparse/cc3m_recaptioned/cc3m-recaptioned-000000.tar"

# 2. SET THE DIRECTORY where you want to save the output images.
# Default is the current working directory.
output_dir = "."
os.makedirs(output_dir, exist_ok=True)


# --- Main Script Logic ---
if not os.path.exists(tar_path):
    print(f"Error: File not found at '{tar_path}'")
    print("Please update the 'tar_path' variable to point to a valid .tar file.")
else:
    print(f"Inspecting first sample from: {os.path.basename(tar_path)}\n")
    dataset = wds.WebDataset(tar_path).decode("pil")
    
    try:
        sample = next(iter(dataset))
        print("Found sample with keys:", list(sample.keys()))

        for k in sample.keys():
            print(sample[k])

        image = None
        image_ext = None
        for ext in ["jpg", "jpeg", "png", "webp"]:
            if ext in sample:
                image = sample[ext]
                image_ext = ext
                break
        
        text_content = None
        text_ext = None
        for ext in ["json", "txt"]:
            if ext in sample:
                text_content = sample[ext]
                text_ext = ext
                break

        pdb.set_trace()

        if image:
            # --- Create the figure ---
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image)
            ax.axis("off")
            
            caption_text = "Image (No text file found)"
            if text_content:
                if text_ext == "json":
                    caption_text = text_content.get("short_caption") or text_content.get("caption", "No caption key found in JSON")
                else:
                    caption_text = text_content
            
            # Set the title, which acts as the caption
            fig.suptitle(caption_text, wrap=True, fontsize=10)

            # --- Save the figure ---
            sample_key = sample.get('__key__', 'unknown_sample')
            output_filename = f"sample_{sample_key}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Use bbox_inches='tight' to minimize whitespace around the image
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig) # Close the figure to free up memory
            
            print(f"\n--- Image (from '.{image_ext}' key) ---")
            print(f"✅ Figure saved to: {output_path}")

        if text_content:
            print(f"\n--- Text Content (from '.{text_ext}' key) ---")
            if text_ext == "json":
                print(json.dumps(text_content, indent=2))
            else:
                print(text_content)
        
        if not image and not text_content:
            print("\nCould not find any standard image or text keys in the sample.")

    except StopIteration:
        print("The .tar file is empty or does not contain any valid samples.")
    except Exception as e:
        print(f"An error occurred: {e}")