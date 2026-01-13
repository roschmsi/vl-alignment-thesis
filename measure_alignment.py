import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

BASE_DIR = "/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data"
IMAGE_EMB_DIR = os.path.join(BASE_DIR, "image_embedding")
TEXT_EMB_DIR = os.path.join(BASE_DIR, "text_embedding")

NUM_SAMPLES = 1024
SEED = 42
TOP_K = 10
QUANTILE = 0.95
H5_KEY = "embeddings"


def remove_outliers(feats, q, exact=False):
    if q == 1:
        return feats
    if exact:
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()
    return feats.clamp(-q_val, q_val)


def prepare_features(feats, q=0.95, exact=False):
    """Outlier removal + Normalization + Move to GPU"""
    if isinstance(feats, torch.Tensor):
        feats = remove_outliers(feats.float(), q=q, exact=exact)
        if torch.cuda.is_available():
            feats = feats.cuda()
        return F.normalize(feats, p=2, dim=-1)
    return feats


def compute_nearest_neighbors(feats, topk=1):
    # feats: (N, D) normalized
    sim = feats @ feats.T
    sim.fill_diagonal_(-float("inf"))
    return sim.argsort(dim=1, descending=True)[:, :topk]


class AlignmentMetrics:
    @staticmethod
    def mutual_knn(feats_A, feats_B, topk):
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)

        n, k = knn_A.shape
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        mask_A = torch.zeros(n, n, device=knn_A.device)
        mask_B = torch.zeros(n, n, device=knn_A.device)

        mask_A[range_tensor, knn_A] = 1.0
        mask_B[range_tensor, knn_B] = 1.0

        acc = (mask_A * mask_B).sum(dim=1) / k
        return acc.mean().item()


def load_fixed_sample(path, num_samples, seed, h5_key="embeddings"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with h5py.File(path, "r") as f:
        if h5_key not in f:
            raise KeyError(
                f"Key '{h5_key}' not found in {path}. Keys: {list(f.keys())}"
            )

        dset = f[h5_key]
        total_len = dset.shape[0]

        rng = np.random.RandomState(seed)
        if total_len < num_samples:
            indices = np.arange(total_len)
        else:
            indices = rng.choice(total_len, size=num_samples, replace=False)

        indices = np.sort(indices)
        data = dset[indices]

    return torch.from_numpy(data)


def find_dataset_file(model_dir):
    candidates = ["cc3m_concat.h5", "cc3m_raw_caption.h5"]

    for fname in candidates:
        full_path = os.path.join(model_dir, fname)
        if os.path.exists(full_path):
            return full_path

    return None


def main():
    print(f"Scanning for models in {BASE_DIR}...")

    vision_models = sorted(
        [
            d
            for d in os.listdir(IMAGE_EMB_DIR)
            if os.path.isdir(os.path.join(IMAGE_EMB_DIR, d))
        ]
    )
    text_models = sorted(
        [
            d
            for d in os.listdir(TEXT_EMB_DIR)
            if os.path.isdir(os.path.join(TEXT_EMB_DIR, d))
        ]
    )

    print(f"Found {len(vision_models)} Vision Models: {vision_models}")
    print(f"Found {len(text_models)} Text Models: {text_models}")

    vision_feats = {}
    print("Loading Vision Features...")
    for model in tqdm(vision_models):
        path = find_dataset_file(os.path.join(IMAGE_EMB_DIR, model))
        if path:
            try:
                raw_feats = load_fixed_sample(path, NUM_SAMPLES, SEED, H5_KEY)
                vision_feats[model] = prepare_features(raw_feats, q=QUANTILE)
            except Exception as e:
                print(f"Failed to load {model}: {e}")
        else:
            print(f"Skipping {model}: No valid cc3m .h5 file found.")

    text_feats = {}
    print("Loading Text Features...")
    for model in tqdm(text_models):
        path = find_dataset_file(os.path.join(TEXT_EMB_DIR, model))
        if path:
            try:
                raw_feats = load_fixed_sample(path, NUM_SAMPLES, SEED, H5_KEY)
                text_feats[model] = prepare_features(raw_feats, q=QUANTILE)
            except Exception as e:
                print(f"Failed to load {model}: {e}")
        else:
            print(f"Skipping {model}: No valid cc3m .h5 file found.")

    results = []

    print("Compute mutual kNN for all combinations...")

    grid_data = np.zeros((len(vision_feats), len(text_feats)))
    v_names = list(vision_feats.keys())
    t_names = list(text_feats.keys())

    for i, v_name in enumerate(v_names):
        for j, t_name in enumerate(t_names):
            score = AlignmentMetrics.mutual_knn(
                vision_feats[v_name], text_feats[t_name], topk=TOP_K
            )

            results.append(
                {"Vision Model": v_name, "Text Model": t_name, "Score": score}
            )
            grid_data[i, j] = score

    print(f"Mutual kNN with k={TOP_K} and N={NUM_SAMPLES})")

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    print(df.to_string(index=False))

    matrix_df = pd.DataFrame(grid_data, index=v_names, columns=t_names)
    print(matrix_df.to_string())


if __name__ == "__main__":
    main()
