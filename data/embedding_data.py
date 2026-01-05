import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import bisect


class H5Base(Dataset):
    def __init__(self, paths, h5_key, indices):
        super().__init__()
        self.indices = np.array(indices)

        total_file_length = 0
        file_boundaries = []
        feature_dim = None
        dtype = None

        print(f"Scanning {len(paths)} files for metadata...")
        for path in paths:
            try:
                with h5py.File(path, "r") as f:
                    dset = f[h5_key]
                    length = len(dset)

                    if feature_dim is None:
                        shape = dset.shape
                        feature_dim = shape[1] if len(shape) > 1 else 1
                        dtype = torch.float32

                    start = total_file_length
                    end = total_file_length + length
                    file_boundaries.append((path, start, end))
                    total_file_length += length
            except Exception as e:
                print(f"Error scanning {path}: {e}")

        # preallocate memory
        print(f"Allocating memory for {len(self.indices)} samples (dim={feature_dim})")
        self.data = torch.zeros((len(self.indices), feature_dim), dtype=dtype)

        # load data from files in slices
        print(f"Loading filtered data from {len(paths)} files")

        indices = self.indices

        for path, file_start, file_end in file_boundaries:
            # Which requested global indices fall into this file
            mask = (indices >= file_start) & (indices < file_end)
            if not np.any(mask):
                continue

            # Global indices we need from this file
            file_global = indices[mask]
            dest_slots = np.where(mask)[0]
            file_local = file_global - file_start  # per-file local indices

            # Sort by local index so we can read contiguous ranges
            order = np.argsort(file_local)
            file_local_sorted = file_local[order]
            dest_sorted = dest_slots[order]

            diffs = np.diff(file_local_sorted)
            run_starts = np.concatenate(([0], np.where(diffs != 1)[0] + 1))
            run_ends = np.concatenate((run_starts[1:], [len(file_local_sorted)]))

            with h5py.File(path, "r") as f:
                dset = f[h5_key]

                for rs, re in tqdm(zip(run_starts, run_ends)):
                    # Local indices for this run
                    run_local = file_local_sorted[rs:re]
                    run_dest = dest_sorted[rs:re]

                    start_idx = int(run_local[0])
                    end_idx = int(run_local[-1]) + 1  # slice end is exclusive

                    # ONE contiguous read from disk
                    chunk_np = dset[
                        start_idx:end_idx
                    ]  # shape: (end_idx-start_idx, feature_dim)

                    # If dtype already matches, avoid unnecessary conversion
                    if dtype == torch.float32 and np.issubdtype(
                        chunk_np.dtype, np.floating
                    ):
                        chunk_torch = torch.from_numpy(chunk_np).to(dtype=dtype)
                    else:
                        # Fallback (may incur a copy)
                        chunk_torch = torch.as_tensor(chunk_np, dtype=dtype)

                    # run_local is consecutive, so chunk_np is already in correct order
                    # The run_dest is in the same order as run_local, so we can assign directly
                    self.data[run_dest] = chunk_torch

        print(f"Successfully loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class H5BimodalDataset(Dataset):
    def __init__(self, text_paths, image_paths, indices, h5_key="embeddings"):
        self.indices = indices
        self.text_db = H5Base(paths=text_paths, h5_key=h5_key, indices=indices)
        self.image_db = H5Base(paths=image_paths, h5_key=h5_key, indices=indices)

    def __len__(self):
        return len(self.text_db)

    def __getitem__(self, idx):
        return self.text_db[idx], self.image_db[idx]


class H5UnimodalDataset(H5Base):
    def __init__(self, paths, indices, h5_key="embeddings"):
        super().__init__(paths=paths, h5_key=h5_key, indices=indices)

    def __getitem__(self, idx):
        return self.data[idx]


class DiffusionDBTextDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with a 'prompt' column.
        """
        print(f"Loading data from {csv_file}...")
        # Read the CSV we just created
        self.df = pd.read_csv(csv_file)

        # Ensure we don't have any empty strings (drop NaNs if any slipped in)
        self.df = self.df.dropna(subset=["prompt"])
        print(f"Dataset ready: {len(self.df)} prompts.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Return the raw text string at the specified index
        return self.df.iloc[idx]["prompt"]


def load_h5_concat_to_torch_fp16(paths, h5_key="embeddings", chunk_rows=262144):
    """Concatenate multiple H5 shards into one torch.float16 tensor. Global id == row index."""
    total = 0
    dim = None
    for p in paths:
        with h5py.File(p, "r") as f:
            d = f[h5_key]
            total += d.shape[0]
            if dim is None:
                dim = d.shape[1]

    out = torch.empty((total, dim), dtype=torch.float16)
    cursor = 0

    for p in paths:
        with h5py.File(p, "r") as f:
            d = f[h5_key]
            n = d.shape[0]
            for i in tqdm(range(0, n, chunk_rows)):
                j = min(i + chunk_rows, n)
                arr = d[i:j].astype(np.float16, copy=False)
                out[cursor + i : cursor + j] = torch.from_numpy(arr)
        cursor += n

    return out


class H5BimodalDatasetWithNNPositives(Dataset):
    """
    Returns a 4-tuple:
      t_anchor, v_anchor, t_pos, v_pos

    Dtypes:
      embeddings are torch.float16

    Shapes after default collate (when num_pos_* > 0):
      t_anchor: (B, Dt)
      v_anchor: (B, Dv)
      t_pos:    (B, Pt, Dt)  or None if num_pos_text==0
      v_pos:    (B, Pv, Dv)  or None if num_pos_vision==0
    """

    def __init__(
        self,
        text_paths,
        image_paths,
        sup_indices,
        text_neighbors_path,
        image_neighbors_path,
        h5_key="embeddings",
        text_topk=5,
        image_topk=5,
        text_nn_positives=1,
        image_nn_positives=1,
        chunk_rows=262144,
    ):
        super().__init__()
        self.sup_indices = np.asarray(sup_indices, dtype=np.int64)

        self.num_pos_text = int(text_nn_positives)
        self.num_pos_vision = int(image_nn_positives)
        self.topk_text = int(text_topk)
        self.topk_vision = int(image_topk)

        if self.num_pos_text < 0 or self.num_pos_vision < 0:
            raise ValueError("num_pos_text/num_pos_vision must be >= 0")
        if (self.num_pos_text > 0 and self.topk_text < 1) or (
            self.num_pos_vision > 0 and self.topk_vision < 1
        ):
            raise ValueError(
                "topk_text/topk_vision must be >= 1 when corresponding num_pos > 0"
            )

        self.text_all = load_h5_concat_to_torch_fp16(
            text_paths, h5_key=h5_key, chunk_rows=chunk_rows
        )
        self.vision_all = load_h5_concat_to_torch_fp16(
            image_paths, h5_key=h5_key, chunk_rows=chunk_rows
        )

        self.text_nn = np.load(text_neighbors_path, mmap_mode="r")  # (N, Kt)
        self.vision_nn = np.load(image_neighbors_path, mmap_mode="r")  # (N, Kv)

        N = self.text_all.shape[0]
        if (
            self.vision_all.shape[0] != N
            or self.text_nn.shape[0] != N
            or self.vision_nn.shape[0] != N
        ):
            raise ValueError(
                "Mismatch in N between embedding banks and neighbor files. "
                f"text_all={self.text_all.shape[0]}, vision_all={self.vision_all.shape[0]}, "
                f"text_nn={self.text_nn.shape[0]}, vision_nn={self.vision_nn.shape[0]}"
            )

        self._rng = None

    def _rng_np(self):
        if self._rng is None:
            seed = torch.initial_seed() % (2**32)
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __len__(self):
        return len(self.sup_indices)

    def _sample_from_topk(
        self, nn_mat, anchor_gid: int, topk: int, num_pos: int
    ) -> np.ndarray:
        K = nn_mat.shape[1]
        pool = min(topk, K)
        cand = nn_mat[anchor_gid, :pool].astype(np.int64)

        rng = self._rng_np()

        if num_pos == 1:
            return np.array(
                [int(cand[int(rng.integers(0, cand.size))])], dtype=np.int64
            )

        if num_pos <= cand.size:
            picks = rng.choice(cand.size, size=num_pos, replace=False)
        else:
            picks = rng.integers(0, cand.size, size=num_pos)

        return cand[picks].astype(np.int64)

    def __getitem__(self, idx):
        anchor_gid = int(self.sup_indices[idx])

        t_anchor = self.text_all[anchor_gid]
        v_anchor = self.vision_all[anchor_gid]

        if self.num_pos_text == 0:
            t_pos = None
        else:
            t_pos_ids = self._sample_from_topk(
                self.text_nn, anchor_gid, self.topk_text, self.num_pos_text
            )
            t_pos = self.text_all[
                torch.as_tensor(t_pos_ids, dtype=torch.long)
            ]  # (Pt, Dt)
            t_pos = t_pos.squeeze() if self.num_pos_text == 1 else t_pos
            # TODO fix this later in case we ever wanna try multiple positives per iteration

        if self.num_pos_vision == 0:
            v_pos = None
        else:
            v_pos_ids = self._sample_from_topk(
                self.vision_nn, anchor_gid, self.topk_vision, self.num_pos_vision
            )
            v_pos = self.vision_all[
                torch.as_tensor(v_pos_ids, dtype=torch.long)
            ]  # (Pv, Dv)
            v_pos = v_pos.squeeze() if self.num_pos_vision == 1 else v_pos
            # TODO fix this later in case we ever wanna try multiple positives per iteration

        return t_anchor, v_anchor, t_pos, v_pos
