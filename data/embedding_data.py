import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import numpy as np
import torch
from tqdm import tqdm


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
