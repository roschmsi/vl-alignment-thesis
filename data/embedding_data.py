import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
from tqdm import tqdm
import glob
from natsort import natsorted
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import h5py
from bisect import bisect_right
import bisect
import torch.distributed as dist
from typing import List, Optional, Tuple, Iterator
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist


def custom_collate_fn(batch):
    if len(batch[0]) == 3:
        text_vectors, image_vectors, extra_text_vectors = zip(*batch)
        return (
            torch.stack(text_vectors, 0),
            torch.stack(image_vectors, 0),
            torch.stack(extra_text_vectors, 0),
        )
    else:
        text_vectors, image_vectors = zip(*batch)
        return torch.stack(text_vectors, 0), torch.stack(image_vectors, 0)

    # text_vectors = pad_sequence(text_vectors, batch_first=True, padding_value=0)
    # image_vectors = pad_sequence(image_vectors, batch_first=True, padding_value=0)

    # if extra_text_vectors:
    #     extra_text_vectors = pad_sequence(extra_text_vectors, batch_first=True, padding_value=0)
    #     return text_vectors, image_vectors, extra_text_vectors
    # else:
    #     return text_vectors, image_vectors


def extract_hidden_states(features, hidden_states_idx=None):
    if hidden_states_idx is None:
        n_layers = features.shape[-1]
        third = n_layers // 3
        idx1 = third
        idx2 = 2 * third
        hidden_states_idx = [idx1, idx2, -1]

    features = features[:, hidden_states_idx]
    features = features.flatten()

    return features


def load_vectors(
    embedding_dirs: list[str],
    hidden_states: bool = False,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    files = []
    for d in embedding_dirs:
        files.extend(natsorted(glob.glob(os.path.join(d, "*.pt"))))

    # TODO remove for full dataset
    # files = files[:2000]

    chunks = []
    with torch.no_grad():
        for file in tqdm(files, desc="Loading embedding data", unit="file"):
            x = torch.load(
                file, map_location="cpu", weights_only=True
            )  # [B, L, D] or [B, D]
            # x = extract_hidden_states(x) if hidden_states else x[..., -1]
            if x.dtype is not dtype:
                x = x.to(dtype)
            # x = x.contiguous()
            x = x.clone()
            chunks.append(x)
    return torch.cat(chunks, dim=0)


class VLEmbeddingDataset(Dataset):
    def __init__(
        self,
        text_embedding_list,
        image_embedding_list,
        extra_text_embedding_list=None,
        train_num_samples=None,
        hidden_states=False,
    ):

        self.text_vectors, self.image_vectors = self._load_image_text_vectors(
            image_embedding_list, text_embedding_list, hidden_states
        )
        n_img, n_txt = len(self.image_vectors), len(self.text_vectors)
        assert (
            n_img > 0 and n_txt > 0 and n_txt % n_img == 0
        ), f"text vectors length ({n_txt}) is not a multiple of image vectors length ({n_img})"

        if extra_text_embedding_list:
            print(f"Loading extra text vectors from {extra_text_embedding_list}")
            self.extra_text_vectors, _ = self._load_image_text_vectors(
                text_embedding_list=extra_text_embedding_list,
                hidden_states=hidden_states,
            )
            assert len(self.extra_text_vectors) == len(
                self.text_vectors
            ), f"extra text vectors length {len(self.extra_text_vectors)} is not equal to text vectors length {len(self.text_vectors)}"

        if train_num_samples is not None:
            num_samples = len(self.text_vectors)
            random_indices = np.random.choice(
                num_samples, train_num_samples, replace=False
            )
            self.text_vectors = [self.text_vectors[i] for i in random_indices]
            self.image_vectors = [self.image_vectors[i] for i in random_indices]
            if extra_text_embedding_list:
                self.extra_text_vectors = [
                    self.extra_text_vectors[i] for i in random_indices
                ]
            print(f"Random Selecting {train_num_samples} samples as training data")

        self.image_num = len(self.image_vectors)
        self.text_num = len(self.text_vectors)

        self.visual_dim = self.image_vectors[0].shape[0]
        self.text_dim = self.text_vectors[0].shape[0]

    def _load_image_text_vectors(
        self, image_embedding_list=None, text_embedding_list=None, hidden_states=False
    ):
        assert (
            image_embedding_list is not None or text_embedding_list is not None
        ), "Either image_embedding_list or text_embedding_list must be provided"
        if image_embedding_list is not None:
            image_vectors = load_vectors(image_embedding_list, hidden_states)
        else:
            image_vectors = []
        if text_embedding_list is not None:
            text_vectors = load_vectors(text_embedding_list, hidden_states)
        else:
            text_vectors = []
        return text_vectors, image_vectors

    def __len__(self):
        return self.text_num

    def __getitem__(self, idx):
        # multiple text for one image
        if idx >= self.image_num:
            img_idx = idx % self.image_num
        else:
            img_idx = idx

        if hasattr(self, "extra_text_vectors"):
            return (
                self.text_vectors[idx],
                self.image_vectors[img_idx],
                self.extra_text_vectors[idx],
            )
        else:
            return self.text_vectors[idx], self.image_vectors[img_idx]


class MMAPDataset(Dataset):
    def __init__(
        self,
        text_embedding_list,
        image_embedding_list,
        extra_text_embedding_list,
        metadata_path=None,
        train_num_samples=None,
        hidden_states=False,
        hidden_states_img_idx=None,
        hidden_states_text_idx=None,
    ):
        super().__init__()

        metadata = torch.load(metadata_path)
        self.num_samples = metadata["num_samples"]
        vision_shape = tuple(metadata["vision_shape"])
        text_shape = tuple(metadata["text_shape"])
        vision_dtype = np.dtype(metadata["vision_dtype"])
        text_dtype = np.dtype(metadata["text_dtype"])

        self.hidden_states = hidden_states
        self.hidden_states_text_idx = hidden_states_text_idx
        self.hidden_states_img_idx = hidden_states_img_idx

        # TODO adapt dealing with list when CC3M, CC12M, YFCC15M are merged
        self.image_vectors = np.memmap(
            image_embedding_list[0],
            dtype=vision_dtype,
            mode="r",
            shape=(self.num_samples, *vision_shape),
        )
        self.text_vectors = np.memmap(
            text_embedding_list[0],
            dtype=text_dtype,
            mode="r",
            shape=(self.num_samples, *text_shape),
        )

        n_img, n_txt = len(self.image_vectors), len(self.text_vectors)
        assert (
            n_img > 0 and n_txt > 0 and n_txt % n_img == 0
        ), f"text vectors length ({n_txt}) is not a multiple of image vectors length ({n_img})"

        if extra_text_embedding_list:
            self.extra_text_vectors = np.memmap(
                extra_text_embedding_list[0],
                dtype=text_dtype,
                mode="r",
                shape=(self.num_samples, *text_shape),
            )

            assert len(self.extra_text_vectors) == len(
                self.text_vectors
            ), f"extra text vectors length {len(self.extra_text_vectors)} is not equal to text vectors length {len(self.text_vectors)}"

        self.image_num = len(self.image_vectors)
        self.text_num = len(self.text_vectors)

        self.visual_dim = self.image_vectors.shape[1]
        self.text_dim = self.text_vectors.shape[1]

        if self.hidden_states:
            if self.hidden_states_img_idx:
                self.visual_dim = self.visual_dim * len(self.hidden_states_img_idx)
            else:
                self.visual_dim = self.visual_dim * 3

            if self.hidden_states_text_idx:
                self.text_dim = self.text_dim * len(self.hidden_states_text_idx)
            else:
                self.text_dim = self.text_dim * 3

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.ascontiguousarray(self.image_vectors[idx])
        txt = np.ascontiguousarray(self.text_vectors[idx])

        if self.hidden_states:
            img = extract_hidden_states(img, self.hidden_states_img_idx)
            txt = extract_hidden_states(txt, self.hidden_states_text_idx)
        else:
            img = img[:, -1]
            txt = txt[:, -1]

        if hasattr(self, "extra_text_vectors"):
            extra = np.ascontiguousarray(self.extra_text_vectors[idx])
            if self.hidden_states:
                extra = extract_hidden_states(extra, self.hidden_states_text_idx)
            else:
                extra = extra[:, -1]

            return torch.from_numpy(txt), torch.from_numpy(img), torch.from_numpy(extra)

        return torch.from_numpy(txt), torch.from_numpy(img)


class H5EmbeddingDataset(Dataset):
    def __init__(
        self,
        text_embedding_list,
        image_embedding_list,
        extra_text_embedding_list,
        hidden_states=False,
        hidden_states_img_idx=None,
        hidden_states_text_idx=None,
        h5_key: str = "embeddings",
    ):
        super().__init__()

        self.hidden_states = hidden_states
        self.hidden_states_text_idx = hidden_states_text_idx
        self.hidden_states_img_idx = hidden_states_img_idx
        self.h5_key = h5_key

        self.text_paths = list(text_embedding_list or [])
        self.image_paths = list(image_embedding_list or [])
        self.extra_text_paths = list(extra_text_embedding_list or [])

        assert (
            len(self.text_paths) > 0
        ), "text_embedding_list must contain at least one .h5 path"
        assert (
            len(self.image_paths) > 0
        ), "image_embedding_list must contain at least one .h5 path"

        if self.extra_text_paths:
            assert len(self.extra_text_paths) > 0

        def _scan(paths):
            lens, shapes = [], []
            for p in paths:
                with h5py.File(p, "r") as f:
                    d = f[self.h5_key]
                    lens.append(d.shape[0])
                    shapes.append(d.shape[1:])
            return lens, shapes

        txt_lens, txt_shapes = _scan(self.text_paths)
        img_lens, img_shapes = _scan(self.image_paths)
        if self.extra_text_paths:
            xt_lens, xt_shapes = _scan(self.extra_text_paths)
        else:
            xt_lens, xt_shapes = None, None

        def _all_equal(xs):
            return all(x == xs[0] for x in xs)

        assert _all_equal(
            txt_shapes
        ), f"text feature shapes differ across files: {txt_shapes}"
        assert _all_equal(
            img_shapes
        ), f"image feature shapes differ across files: {img_shapes}"

        if self.extra_text_paths:
            assert _all_equal(
                xt_shapes
            ), f"extra-text feature shapes differ across files: {xt_shapes}"

        self._txt_cumlens = np.cumsum([0] + txt_lens)
        self._img_cumlens = np.cumsum([0] + img_lens)
        if self.extra_text_paths:
            self._xt_cumlens = np.cumsum([0] + xt_lens)
        else:
            self._xt_cumlens = None

        txt_total = int(self._txt_cumlens[-1])
        img_total = int(self._img_cumlens[-1])
        xt_total = int(self._xt_cumlens[-1]) if self._xt_cumlens is not None else None

        # Alignment assumption: 1:1 across modalities at the sample level
        # If your text is a multiple of images (repeated image rows), build files that already reflect that.
        assert (
            txt_total == img_total
        ), f"Global lengths differ (text={txt_total}, image={img_total}). Make them aligned."
        if xt_total is not None:
            assert (
                xt_total == txt_total
            ), f"Global lengths differ (extra_text={xt_total}, text={txt_total})."

        self.num_samples = txt_total
        vision_shape = img_shapes[0]
        text_shape = txt_shapes[0]

        self.image_num = img_total
        self.text_num = txt_total
        self.visual_dim = vision_shape[0] if len(vision_shape) >= 1 else 1
        self.text_dim = text_shape[0] if len(text_shape) >= 1 else 1

        if self.hidden_states:
            if self.hidden_states_img_idx:
                self.visual_dim *= len(self.hidden_states_img_idx)
            else:
                self.visual_dim *= 3
            if self.hidden_states_text_idx:
                self.text_dim *= len(self.hidden_states_text_idx)
            else:
                self.text_dim *= 3

        # Lazy-open file handles/datasets (per worker)
        self._txt_files = [None] * len(self.text_paths)
        self._img_files = [None] * len(self.image_paths)
        self._xt_files = (
            [None] * len(self.extra_text_paths) if self.extra_text_paths else None
        )

        self._txt_dsets = [None] * len(self.text_paths)
        self._img_dsets = [None] * len(self.image_paths)
        self._xt_dsets = (
            [None] * len(self.extra_text_paths) if self.extra_text_paths else None
        )

    def __len__(self):
        return self.num_samples

    # ---- utils ----
    @staticmethod
    def _locate(cumlens, idx):
        # find file j such that cumlens[j] ≤ idx < cumlens[j+1]
        j = int(bisect_right(cumlens, idx) - 1)
        local = idx - int(cumlens[j])
        return j, local

    def _ensure_open(self, which: str, j: int):
        if which == "txt":
            if self._txt_dsets[j] is None:
                f = h5py.File(self.text_paths[j], "r")
                self._txt_files[j] = f
                self._txt_dsets[j] = f[self.h5_key]
        elif which == "img":
            if self._img_dsets[j] is None:
                f = h5py.File(self.image_paths[j], "r")
                self._img_files[j] = f
                self._img_dsets[j] = f[self.h5_key]
        elif which == "xt":
            if self._xt_dsets[j] is None:
                f = h5py.File(self.extra_text_paths[j], "r")
                self._xt_files[j] = f
                self._xt_dsets[j] = f[self.h5_key]
        else:
            raise ValueError(which)

    def __getitem__(self, idx):
        # Locate in each modality using the same global index
        j_txt, i_txt = self._locate(self._txt_cumlens, idx)
        self._ensure_open("txt", j_txt)
        txt = np.ascontiguousarray(self._txt_dsets[j_txt][i_txt])

        j_img, i_img = self._locate(self._img_cumlens, idx)
        self._ensure_open("img", j_img)
        img = np.ascontiguousarray(self._img_dsets[j_img][i_img])

        if self.hidden_states:
            img = extract_hidden_states(img, self.hidden_states_img_idx)
            txt = extract_hidden_states(txt, self.hidden_states_text_idx)

        if self.extra_text_paths:
            j_xt, i_xt = self._locate(self._xt_cumlens, idx)
            self._ensure_open("xt", j_xt)
            extra = np.ascontiguousarray(self._xt_dsets[j_xt][i_xt])

            if self.hidden_states:
                extra = extract_hidden_states(extra, self.hidden_states_text_idx)
            # else:
            #     extra = extra[:, -1]

            return torch.from_numpy(txt), torch.from_numpy(img), torch.from_numpy(extra)

        return torch.from_numpy(txt), torch.from_numpy(img)

    def close(self):
        for arr in (self._txt_files, self._img_files, self._xt_files or []):
            for f in arr:
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass

    def __del__(self):
        self.close()


class H5EmbeddingIterableDataset(IterableDataset):
    """
    WebDataset-style streaming over (possibly many) HDF5 files:
      - Shuffle at the *block/segment* level (cheap permutation).
      - Within a block, read *contiguously* and feed a small *shuffle buffer*.
      - Each worker opens its own HDF5 handles (no sharing).
    """

    def __init__(
        self,
        text_embedding_list: List[str],
        image_embedding_list: List[str],
        extra_text_embedding_list: Optional[List[str]] = None,
        hidden_states: bool = False,
        hidden_states_img_idx: Optional[List[int]] = None,
        hidden_states_text_idx: Optional[List[int]] = None,
        h5_key: str = "embeddings",
        # Streaming/shuffle controls
        block_size: int = 100_000,  # logical “segment” length in samples
        buffer_size: int = 4096,  # in-memory shuffle buffer
        read_batch: int = 4096,  # contiguous read granularity
        seed: int = 0,  # base RNG seed
        per_worker_block_offset: bool = True,  # stagger blocks across workers
        drop_last_incomplete_block: bool = False,
        # HDF5 open kwargs (override as needed)
        h5_open_kwargs: Optional[dict] = None,  # e.g., {"libver":"latest","swmr":True}
    ):
        super().__init__()

        # ---- Store config ----
        self.hidden_states = hidden_states
        self.hidden_states_text_idx = hidden_states_text_idx
        self.hidden_states_img_idx = hidden_states_img_idx
        self.h5_key = h5_key

        self.text_paths = list(text_embedding_list or [])
        self.image_paths = list(image_embedding_list or [])
        self.extra_text_paths = list(extra_text_embedding_list or [])

        assert (
            len(self.text_paths) > 0
        ), "text_embedding_list must contain at least one .h5 path"
        assert (
            len(self.image_paths) > 0
        ), "image_embedding_list must contain at least one .h5 path"
        if self.extra_text_paths:
            assert len(self.extra_text_paths) > 0

        self.block_size = int(block_size)
        self.buffer_size = int(buffer_size)
        self.read_batch = int(read_batch)
        self.seed = int(seed)
        self.per_worker_block_offset = bool(per_worker_block_offset)
        self.drop_last_incomplete_block = bool(drop_last_incomplete_block)
        self.h5_open_kwargs = dict(h5_open_kwargs or {})

        # ---- Probe files to get lengths & shapes (close immediately) ----
        def _scan(paths):
            lens, shapes = [], []
            for p in paths:
                with h5py.File(p, "r") as f:
                    d = f[self.h5_key]
                    lens.append(int(d.shape[0]))
                    shapes.append(tuple(d.shape[1:]))
            return lens, shapes

        txt_lens, txt_shapes = _scan(self.text_paths)
        img_lens, img_shapes = _scan(self.image_paths)
        if self.extra_text_paths:
            xt_lens, xt_shapes = _scan(self.extra_text_paths)
        else:
            xt_lens, xt_shapes = None, None

        def _all_equal(xs):
            return all(x == xs[0] for x in xs)

        assert _all_equal(
            txt_shapes
        ), f"text feature shapes differ across files: {txt_shapes}"
        assert _all_equal(
            img_shapes
        ), f"image feature shapes differ across files: {img_shapes}"
        if self.extra_text_paths:
            assert _all_equal(
                xt_shapes
            ), f"extra-text feature shapes differ across files: {xt_shapes}"

        # Global cumlens to map [0, N) -> (file_idx, local_idx)
        self._txt_cumlens = np.cumsum([0] + txt_lens, dtype=np.int64)
        self._img_cumlens = np.cumsum([0] + img_lens, dtype=np.int64)
        if self.extra_text_paths:
            self._xt_cumlens = np.cumsum([0] + xt_lens, dtype=np.int64)
        else:
            self._xt_cumlens = None

        self.num_samples = int(self._txt_cumlens[-1])
        assert self.num_samples == int(
            self._img_cumlens[-1]
        ), f"Global lengths differ (text={self.num_samples}, image={int(self._img_cumlens[-1])})."
        if self._xt_cumlens is not None:
            assert self.num_samples == int(
                self._xt_cumlens[-1]
            ), f"Global lengths differ (extra_text={int(self._xt_cumlens[-1])}, text={self.num_samples})."

        # expose dims, matching your original API
        self.image_num = self.num_samples
        self.text_num = self.num_samples
        vision_shape = img_shapes[0]
        text_shape = txt_shapes[0]
        self.visual_dim = vision_shape[0] if len(vision_shape) else 1
        self.text_dim = text_shape[0] if len(text_shape) else 1
        if self.hidden_states:
            self.visual_dim *= (
                len(self.hidden_states_img_idx) if self.hidden_states_img_idx else 3
            )
            self.text_dim *= (
                len(self.hidden_states_text_idx) if self.hidden_states_text_idx else 3
            )

        # Lazily-opened per-worker handles
        self._txt_files = [None] * len(self.text_paths)
        self._img_files = [None] * len(self.image_paths)
        self._xt_files = (
            [None] * len(self.extra_text_paths) if self.extra_text_paths else None
        )

        self._txt_dsets = [None] * len(self.text_paths)
        self._img_dsets = [None] * len(self.image_paths)
        self._xt_dsets = (
            [None] * len(self.extra_text_paths) if self.extra_text_paths else None
        )

    # -------- helpers --------
    @staticmethod
    def _locate(cumlens: np.ndarray, idx: int) -> Tuple[int, int]:
        j = int(bisect_right(cumlens, idx) - 1)
        local = int(idx - cumlens[j])
        return j, local

    @staticmethod
    def _is_valid_dset(d):
        try:
            return (d is not None) and hasattr(d, "id") and d.id.valid
        except Exception:
            return False

    def _ensure_open(self, which: str, j: int):
        kwargs = dict(self.h5_open_kwargs)

        if which == "txt":
            dsets, files, paths = self._txt_dsets, self._txt_files, self.text_paths
        elif which == "img":
            dsets, files, paths = self._img_dsets, self._img_files, self.image_paths
        elif which == "xt":
            dsets, files, paths = self._xt_dsets, self._xt_files, self.extra_text_paths
        else:
            raise ValueError(which)

        # Reopen if missing or invalid (closed) handle
        if not self._is_valid_dset(dsets[j]):
            f = h5py.File(paths[j], "r", **kwargs)
            files[j] = f
            dsets[j] = f[self.h5_key]

    def _slice_across_files(
        self, cumlens: np.ndarray, start: int, end: int, which: str  # exclusive
    ) -> Iterator[np.ndarray]:
        """
        Yield contiguous slices [s:e) from the appropriate files so that
        concatenating yields the same as dataset[start:end].
        """
        assert 0 <= start <= end <= int(cumlens[-1])
        s = start
        while s < end:
            j, loc = self._locate(cumlens, s)
            self._ensure_open(which, j)
            dset = (
                self._txt_dsets
                if which == "txt"
                else self._img_dsets if which == "img" else self._xt_dsets
            )[j]
            file_end_global = int(cumlens[j + 1])
            e = min(end, file_end_global)
            yield dset[loc : loc + (e - s)]
            s = e

    def _yield_block(
        self, rng: random.Random, start: int, end: int
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Sequentially read [start, end) in contiguous chunks (read_batch),
        push into a local shuffle buffer, and pop random elements.
        """
        buf = []

        def _drain_to(target_size: int):
            # pop uniformly at random until len(buf) == target_size
            while len(buf) > target_size:
                k = rng.randrange(len(buf))
                yield buf.pop(k)

        # Walk the block in contiguous steps
        cur = start
        while cur < end:
            step = min(self.read_batch, end - cur)
            # Collect slices across files for each modality, then concatenate once
            txt_parts, img_parts, xt_parts = [], [], []

            for arr in self._slice_across_files(
                self._txt_cumlens, cur, cur + step, "txt"
            ):
                txt_parts.append(arr)
            for arr in self._slice_across_files(
                self._img_cumlens, cur, cur + step, "img"
            ):
                img_parts.append(arr)
            if self._xt_cumlens is not None:
                for arr in self._slice_across_files(
                    self._xt_cumlens, cur, cur + step, "xt"
                ):
                    xt_parts.append(arr)

            txt_blk = (
                np.concatenate(txt_parts, axis=0, dtype=txt_parts[0].dtype)
                if len(txt_parts) > 1
                else txt_parts[0]
            )
            img_blk = (
                np.concatenate(img_parts, axis=0, dtype=img_parts[0].dtype)
                if len(img_parts) > 1
                else img_parts[0]
            )
            xt_blk = None
            if self._xt_cumlens is not None:
                xt_blk = (
                    np.concatenate(xt_parts, axis=0, dtype=xt_parts[0].dtype)
                    if len(xt_parts) > 1
                    else xt_parts[0]
                )

            # Per-sample postprocess + push into buffer
            if self.hidden_states:
                # Expect [layers, dim] per sample (your original logic)
                if self.hidden_states_img_idx is not None:
                    img_blk = extract_hidden_states(img_blk, self.hidden_states_img_idx)
                else:
                    img_blk = extract_hidden_states(img_blk, None)
                if self.hidden_states_text_idx is not None:
                    txt_blk = extract_hidden_states(
                        txt_blk, self.hidden_states_text_idx
                    )
                else:
                    txt_blk = extract_hidden_states(txt_blk, None)
                if xt_blk is not None:
                    if self.hidden_states_text_idx is not None:
                        xt_blk = extract_hidden_states(
                            xt_blk, self.hidden_states_text_idx
                        )
                    else:
                        xt_blk = extract_hidden_states(xt_blk, None)
            else:
                # Keep parity with your original: [:, -1]
                img_blk = img_blk  # [..., -1]
                txt_blk = txt_blk  # [..., -1]
                if xt_blk is not None:
                    xt_blk = xt_blk  # [..., -1]

            # Push
            if xt_blk is None:
                for i in range(len(txt_blk)):
                    buf.append(
                        (
                            torch.from_numpy(np.ascontiguousarray(txt_blk[i])),
                            torch.from_numpy(np.ascontiguousarray(img_blk[i])),
                        )
                    )
            else:
                for i in range(len(txt_blk)):
                    buf.append(
                        (
                            torch.from_numpy(np.ascontiguousarray(txt_blk[i])),
                            torch.from_numpy(np.ascontiguousarray(img_blk[i])),
                            torch.from_numpy(np.ascontiguousarray(xt_blk[i])),
                        )
                    )

            # Pop down to buffer_size
            yield from _drain_to(self.buffer_size)
            cur += step

        # Drain the remainder (light shuffle)
        rng.shuffle(buf)
        for item in buf:
            yield item

    # -------- IterableDataset protocol --------
    def set_epoch(self, epoch: int):
        # Optional: call this from your training loop each epoch to reshuffle blocks
        self._epoch = int(epoch)

    def __iter__(self):
        # --- DDP rank/world ---
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1

        # --- Worker info ---
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # --- Combined shard id across ranks and workers ---
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        # --- RNG: include seed, rank, worker, and (optional) epoch ---
        base_seed = int(self.seed)
        epoch = getattr(self, "_epoch", 0)
        rng = random.Random(
            (base_seed ^ (rank + 0x9E3779B9)) ^ (worker_id << 16) ^ (epoch << 8)
        )

        # --- Build block list over [0, N) ---
        N = self.num_samples
        num_blocks = N // self.block_size + (
            0 if (self.drop_last_incomplete_block or N % self.block_size == 0) else 1
        )
        blocks = []
        for b in range(num_blocks):
            s = b * self.block_size
            e = min((b + 1) * self.block_size, N)
            if e - s <= 0:
                continue
            if self.drop_last_incomplete_block and (e - s) < self.block_size:
                continue
            blocks.append((s, e))

        # --- Distribute blocks across ALL shards (ranks × workers) ---
        if num_shards > 1:
            # Each shard gets blocks where index % num_shards == shard_id
            blocks = [
                blk for i, blk in enumerate(blocks) if (i % num_shards) == shard_id
            ]

        # --- Optional offset to skew I/O starts (use shard_id, not just worker_id) ---
        if self.per_worker_block_offset and len(blocks) > 1:
            off = shard_id % len(blocks)
            blocks = blocks[off:] + blocks[:off]

        # --- Shuffle block order per (rank, worker, epoch) ---
        rng.shuffle(blocks)

        try:
            for s, e in blocks:
                yield from self._yield_block(rng, s, e)
        finally:
            self.close()

    def close(self):
        # Close and clear ALL cached refs so a future __iter__ can reopen cleanly
        for files, dsets in (
            (self._txt_files, self._txt_dsets),
            (self._img_files, self._img_dsets),
            ((self._xt_files or []), (self._xt_dsets or [])),
        ):
            if files is None or dsets is None:
                continue
            for i in range(len(files)):
                try:
                    if files[i] is not None:
                        files[i].close()
                except Exception:
                    pass
                # IMPORTANT: drop references so we don’t reuse closed objects
                files[i] = None
                if i < len(dsets):
                    dsets[i] = None

    def __del__(self):
        self.close()

    def __len__(self):
        # number of *samples* produced per full pass of __iter__
        return self.num_samples

    def read_by_global_index(self, gidx: int):
        """
        Fetch a single aligned (text, image[, extra_text]) sample by global index.
        Returns torch tensors ready to use.
        Safe to call from worker threads (uses per-worker handles).
        """
        assert 0 <= gidx < self.num_samples, f"gidx out of range: {gidx}"

        # Locate & open text
        tj, tloc = self._locate(self._txt_cumlens, gidx)
        self._ensure_open("txt", tj)
        txt_np = self._txt_dsets[tj][tloc]

        # Locate & open image
        ij, iloc = self._locate(self._img_cumlens, gidx)
        self._ensure_open("img", ij)
        img_np = self._img_dsets[ij][iloc]

        # Optional extra text
        xt_np = None
        if self._xt_cumlens is not None:
            xj, xloc = self._locate(self._xt_cumlens, gidx)
            self._ensure_open("xt", xj)
            xt_np = self._xt_dsets[xj][xloc]

        # Hidden-states handling (same logic you use in streaming path)
        if self.hidden_states:
            img_np = extract_hidden_states(
                img_np,
                (
                    self.hidden_states_img_idx
                    if self.hidden_states_img_idx is not None
                    else None
                ),
            )
            txt_np = extract_hidden_states(
                txt_np,
                (
                    self.hidden_states_text_idx
                    if self.hidden_states_text_idx is not None
                    else None
                ),
            )
            if xt_np is not None:
                xt_np = extract_hidden_states(
                    xt_np,
                    (
                        self.hidden_states_text_idx
                        if self.hidden_states_text_idx is not None
                        else None
                    ),
                )
        else:
            # keep last-layer or full vector per your current convention (no-op here)
            pass

        # Convert to contiguous tensors
        t = torch.from_numpy(np.ascontiguousarray(txt_np))
        v = torch.from_numpy(np.ascontiguousarray(img_np))
        if xt_np is not None:
            x = torch.from_numpy(np.ascontiguousarray(xt_np))
            return (t, v, x)
        else:
            return (t, v)


def batched_collate_fn(batch):
    """
    Collate function for BatchedLazyDataset.
    It receives a list of (text_batch, image_batch) tuples and concatenates them.
    """
    if len(batch[0]) == 3:
        text_batches, image_batches, extra_text_batches = zip(*batch)
        final_text = torch.cat(text_batches, dim=0)
        final_images = torch.cat(image_batches, dim=0)
        final_extra_text = torch.cat(extra_text_batches, dim=0)
        return final_text, final_images, final_extra_text
    else:
        text_batches, image_batches = zip(*batch)
        final_text = torch.cat(text_batches, dim=0)
        final_images = torch.cat(image_batches, dim=0)
        return final_text, final_images


def batched_collate_fn(batch):
    if len(batch[0]) == 3:
        text_batches, image_batches, extra_text_batches = zip(*batch)
        final_text = torch.cat(text_batches, dim=0)
        final_images = torch.cat(image_batches, dim=0)
        final_extra_text = torch.cat(extra_text_batches, dim=0)
        return final_text, final_images, final_extra_text
    else:
        text_batches, image_batches = zip(*batch)
        final_text = torch.cat(text_batches, dim=0)
        final_images = torch.cat(image_batches, dim=0)
        return final_text, final_images


class BatchedLazyDataset(Dataset):
    """
    I/O-efficient Dataset that loads entire pre-batched files.
    Shuffling is at the file-level.
    """

    def __init__(
        self, text_files, image_files, extra_text_files=None, hidden_states=False
    ):
        self.text_files = text_files
        self.image_files = image_files
        self.extra_text_files = extra_text_files
        self.hidden_states = hidden_states

        assert len(self.text_files) > 0, "No text embedding files found."
        assert len(self.text_files) == len(
            self.image_files
        ), "Mismatch in number of text and image files"
        if self.extra_text_files:
            assert len(self.extra_text_files) == len(
                self.text_files
            ), "Mismatch in number of extra text and text files"

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        text_file_path = self.text_files[idx]
        image_file_path = self.image_files[idx]

        text_batch = torch.load(text_file_path, weights_only=True).to(torch.float16)
        image_batch = torch.load(image_file_path, weights_only=True).to(torch.float16)

        if self.hidden_states:
            text_batch = extract_hidden_states(text_batch)
            image_batch = extract_hidden_states(image_batch)
        else:
            # take the final representation
            text_batch = text_batch[:, :, -1]
            image_batch = image_batch[:, :, -1]

        if self.extra_text_files:
            extra_text_file_path = self.extra_text_files[idx]
            extra_text_batch = torch.load(extra_text_file_path, weights_only=True).to(
                torch.float16
            )
            if self.hidden_states:
                extra_text_batch = extract_hidden_states(extra_text_batch)
            # TODO return the final representation if hidden_states == False
            return text_batch, image_batch, extra_text_batch
        else:
            return text_batch, image_batch

    def get_total_samples(self):
        if len(self.text_files) == 0:
            return 0
        total_samples = (len(self.text_files) - 1) * 32
        total_samples += torch.load(self.text_files[-1]).shape[0]
        # for file_path in tqdm(self.text_files, desc="Calculating total samples"):
        #     tensor_info = torch.load(file_path, map_location='cpu', weights_only=True)
        #     total_samples += tensor_info.shape[0]
        return total_samples

    def get_dimensions(self):
        # Load the first file of each type to determine dimensions
        text_batch_sample = torch.load(
            self.text_files[0], map_location="cpu", weights_only=True
        )
        if self.hidden_states:
            text_batch_sample = extract_hidden_states(text_batch_sample)
        else:
            text_batch_sample = text_batch_sample[:, :, -1]
        text_dim = text_batch_sample.shape[-1]

        image_batch_sample = torch.load(
            self.image_files[0], map_location="cpu", weights_only=True
        )
        if self.hidden_states:
            image_batch_sample = extract_hidden_states(image_batch_sample)
        else:
            image_batch_sample = image_batch_sample[:, :, -1]
        visual_dim = image_batch_sample.shape[-1]

        return visual_dim, text_dim


def build_pairing_plan(num_samples: int, k_supervised: int, seed: int = 0):
    assert 0 < k_supervised <= num_samples
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)

    paired_idx = np.sort(perm[:k_supervised])  # keep aligned
    rest = perm[k_supervised:]

    # independent shuffles for unsupervised split
    unsup_t = rest.copy()
    rng.shuffle(unsup_t)
    unsup_v = rest.copy()
    rng.shuffle(unsup_v)

    if len(unsup_v) > 1:  # reduce accidental matches
        unsup_v = np.roll(unsup_v, 1)

    return paired_idx, unsup_t, unsup_v


class _Shard:
    @staticmethod
    def info():
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        else:
            rank, world = 0, 1
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1
        return rank * nworkers + wid, world * nworkers

    @staticmethod
    def take(indices: np.ndarray):
        sid, nshards = _Shard.info()
        return indices if nshards <= 1 else indices[sid::nshards]


class PairedSubsetDataset(IterableDataset):
    def __init__(self, base, paired_idx):
        super().__init__()
        self.base = base
        self.idx = np.asarray(paired_idx, dtype=np.int64)

    def __len__(self):
        # Report GLOBAL length; do NOT shard here
        return int(self.idx.size)

    def __iter__(self):
        # If you want no worker sharding on single-GPU, just iterate self.idx
        local_idx = self.idx
        # If you DO want worker sharding, do it explicitly here using get_worker_info()
        # and/or torch.distributed, but keep __len__ global.

        try:
            for g in local_idx:
                yield self.base.read_by_global_index(int(g))
        finally:
            self.base.close()


class UnpairedSubsetDataset(IterableDataset):
    """Streams de-aligned pairs by independently choosing text and image indices."""

    def __init__(
        self,
        base: H5EmbeddingIterableDataset,
        text_idx: np.ndarray,
        image_idx: np.ndarray,
        reshuffle_each_epoch: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        assert len(text_idx) == len(image_idx)
        self.base = base
        self.t = np.asarray(text_idx, dtype=np.int64)
        self.v = np.asarray(image_idx, dtype=np.int64)
        self.reshuffle = reshuffle_each_epoch
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, e: int):
        self._epoch = int(e)

    def __iter__(self):
        t = _Shard.take(self.t).copy()
        v = _Shard.take(self.v).copy()

        if self.reshuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(t)
            rng.shuffle(v)
            if len(v) > 1:
                v = np.roll(v, 1)

        try:
            for ti, vi in zip(t, v):
                s_t = self.base.read_by_global_index(int(ti))
                s_v = self.base.read_by_global_index(int(vi))
                if len(s_t) == 2:
                    t_tensor, _ = s_t
                    _, v_tensor = s_v
                    yield (t_tensor, v_tensor)
                else:
                    t_tensor, _, x_tensor = s_t
                    _, v_tensor, _ = s_v
                    yield (t_tensor, v_tensor, x_tensor)  # keep x aligned to TEXT
        finally:
            self.base.close()

    def __len__(self):
        return int(self.t.size)


class _InMemoryFilteredH5Base(Dataset):
    def __init__(self, paths, h5_key, indices):
        super().__init__()
        self.indices = np.array(indices)

        total_file_length = 0
        file_boundaries = []  # Stores (path, start_idx, end_idx)
        feature_dim = None
        dtype = None

        # --- Step 1: Scan files to map boundaries and get dimensions ---
        print(f"Scanning {len(paths)} files for metadata...")
        for path in paths:
            try:
                with h5py.File(path, "r") as f:
                    dset = f[h5_key]
                    length = len(dset)

                    # Capture dimension and dtype from the first valid file
                    if feature_dim is None:
                        shape = dset.shape
                        feature_dim = shape[1] if len(shape) > 1 else 1
                        # We usually want Float32 for training, even if disk is Float16
                        dtype = torch.float32

                    start = total_file_length
                    end = total_file_length + length
                    file_boundaries.append((path, start, end))
                    total_file_length += length
            except Exception as e:
                print(f"Error scanning {path}: {e}")

        # --- Step 2: Pre-allocate the exact memory needed ---
        # We create the final container now.
        # len(self.indices) ensures we only store what we need.
        print(
            f"Allocating memory for {len(self.indices)} samples (dim={feature_dim})..."
        )
        self.data = torch.zeros((len(self.indices), feature_dim), dtype=dtype)

        # --- Step 3: Load specific slices from files (range-based, fewer I/Os) ---
        print(f"Loading filtered data from {len(paths)} files (range-based reads)...")

        indices = self.indices  # just a shorthand

        for path, file_start, file_end in file_boundaries:
            # Which requested global indices fall into this file?
            mask = (indices >= file_start) & (indices < file_end)
            if not np.any(mask):
                continue

            # Global indices we need from this file
            file_global = indices[mask]  # shape: (K,)
            dest_slots = np.where(mask)[0]  # where to place them in self.data
            file_local = file_global - file_start  # convert to per-file indices

            # Sort by local index so we can read contiguous ranges
            order = np.argsort(file_local)
            file_local_sorted = file_local[order]
            dest_sorted = dest_slots[order]

            diffs = np.diff(file_local_sorted)
            run_starts = np.concatenate(([0], np.where(diffs != 1)[0] + 1))
            run_ends = np.concatenate((run_starts[1:], [len(file_local_sorted)]))

            with h5py.File(path, "r") as f:
                dset = f[h5_key]

                for rs, re in zip(run_starts, run_ends):
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

        print(f"Successfully loaded {len(self.data)} filtered samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class H5BimodalDataset(Dataset):
    def __init__(self, text_paths, image_paths, indices, h5_key="embeddings"):
        self.indices = indices
        self.text_db = _InMemoryFilteredH5Base(
            paths=text_paths, h5_key=h5_key, indices=indices
        )
        self.image_db = _InMemoryFilteredH5Base(
            paths=image_paths, h5_key=h5_key, indices=indices
        )

    def __len__(self):
        return len(self.text_db)

    def __getitem__(self, idx):
        return self.text_db[idx], self.image_db[idx]


class H5UnimodalDataset(_InMemoryFilteredH5Base):
    def __init__(self, paths, indices, h5_key="embeddings"):
        super().__init__(paths=paths, h5_key=h5_key, indices=indices)

    def __getitem__(self, idx):
        return self.data[idx]
