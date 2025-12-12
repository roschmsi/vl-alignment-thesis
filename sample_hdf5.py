# save as make_first_n_subset.py
import os
import argparse
import numpy as np
import h5py
import torch

# No need for your Dataset classes here—direct HDF5→HDF5 copy is faster and lighter.


def _map_dtype(name: str | None, src_np_dtype):
    """Map a --dtype string to a NumPy dtype; None => preserve."""
    if name is None or name == "preserve":
        return src_np_dtype
    name = name.lower()
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    if name == "int64":
        return np.int64
    if name == "int32":
        return np.int32
    raise ValueError(f"Unsupported dtype: {name}")


def write_first_n_h5(
    input_path: str,
    output_path: str,
    key: str = "embeddings",
    n: int = 100_000,
    out_dtype: str | None = "float16",
    chunk_rows: int = 32,
    overwrite: bool = False,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(f"{output_path} exists. Use --overwrite to replace.")
        os.remove(output_path)

    with h5py.File(input_path, "r") as fin:
        if key not in fin:
            raise KeyError(f"Key '{key}' not found in {input_path}")
        src = fin[key]
        total = len(src)
        take = min(n, total)

        # Normalize shapes: ensure 2D (N, D...) becomes (N, prod(D...)) for consistency?
        # Here we *preserve* original trailing dims exactly.
        out_shape = (take,) + src.shape[1:]
        target_np_dtype = _map_dtype(out_dtype, src.dtype)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            dset = fout.create_dataset(
                key,
                shape=out_shape,
                dtype=target_np_dtype,
                chunks=(min(chunk_rows, take),) + src.shape[1:] if take > 0 else None,
                compression="gzip",
                compression_opts=1,
            )

            # Chunked copy of the FIRST N rows
            for s in range(0, take, chunk_rows):
                e = min(s + chunk_rows, take)
                chunk = src[s:e]  # numpy view
                if chunk.dtype != target_np_dtype:
                    chunk = chunk.astype(target_np_dtype, copy=False)
                dset[s:e] = chunk

    print(f"Wrote first {take} rows from {input_path} -> {output_path} (key='{key}')")


def parse_args():
    p = argparse.ArgumentParser(
        description="Write the FIRST N rows from a single HDF5 file into a new HDF5."
    )
    p.add_argument("--input", required=True, help="Input .h5 path (single file).")
    p.add_argument("--out", required=True, help="Output .h5 path.")
    p.add_argument("--key", default="embeddings", help="HDF5 dataset key to copy.")
    p.add_argument(
        "--n", type=int, default=100000, help="Number of leading rows to take."
    )
    p.add_argument(
        "--dtype",
        default="float16",
        help='Output dtype: "preserve", "float32", "float64", "int64", "int32".',
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output."
    )
    p.add_argument("--chunk-rows", type=int, default=32, help="Row chunk size for I/O.")
    return p.parse_args()


def main():
    args = parse_args()
    write_first_n_h5(
        input_path=args.input,
        output_path=args.out,
        key=args.key,
        n=args.n,
        out_dtype=args.dtype,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
