import torch
import numpy as np
import faiss
import h5py
import os
import argparse
import time
from tqdm import tqdm


def get_total_size(file_list, key):
    total = 0
    for p in file_list:
        with h5py.File(p, "r") as f:
            total += f[key].shape[0]
    return total


def iterate_hdf5_chunks(file_list, key, batch_size):
    for p in file_list:
        print(f"Reading file: {p}")
        with h5py.File(p, "r") as f:
            dset = f[key]
            total_rows = dset.shape[0]

            for i in range(0, total_rows, batch_size):
                end = min(i + batch_size, total_rows)
                chunk = dset[i:end]
                yield chunk


def generate_neighbor_indices(
    query_file_list,
    output_path,
    index_file_list=None,
    k=64,
    batch_size=65536,
    h5_key="embeddings",
    fast_mode=False,
    use_gpu=False,
):
    if index_file_list is None:
        index_file_list = query_file_list
        is_cross_modal = False
    else:
        is_cross_modal = True

    with h5py.File(index_file_list[0], "r") as f:
        dim = f[h5_key].shape[1]

    total_db_samples = get_total_size(index_file_list, h5_key)
    print(f"Database: {total_db_samples} samples | Dimension: {dim}")

    nlist = 4096
    quantizer = faiss.IndexFlatIP(dim)

    if fast_mode:
        m = 64
        if dim % m != 0:
            m = 32
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT
        )
    else:
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    if use_gpu:
        print("Moving Index to GPU...")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.useFloat16LookupTables = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
    else:
        faiss.omp_set_num_threads(min(os.cpu_count(), 32))

    print("Training Index ...")
    start_t = time.time()
    train_vectors = []
    needed_samples = 262144

    db_iter = iterate_hdf5_chunks(index_file_list, h5_key, batch_size)

    for chunk in db_iter:
        batch_np = chunk.astype("float32")
        faiss.normalize_L2(batch_np)
        train_vectors.append(batch_np)
        if len(train_vectors) * batch_size >= needed_samples:
            break

    if len(train_vectors) > 0:
        train_vectors = np.concatenate(train_vectors, axis=0)
        index.train(train_vectors)
        del train_vectors
    print(f"Index trained in {time.time()-start_t:.1f}s.")

    print(f"Adding {total_db_samples} vectors ...")
    start_t = time.time()

    db_iter = iterate_hdf5_chunks(index_file_list, h5_key, batch_size)

    for chunk in tqdm(
        db_iter, total=(total_db_samples // batch_size) + 1, desc="Indexing"
    ):
        batch_np = chunk.astype("float32")
        faiss.normalize_L2(batch_np)
        index.add(batch_np)

    print(f"Indexing completed in {time.time()-start_t:.1f}s.")

    print(f"Searching for neighbors ...")
    index.nprobe = 16
    all_indices = []

    search_k = k + 1 if not is_cross_modal else k
    global_idx = 0

    query_iter = iterate_hdf5_chunks(query_file_list, h5_key, batch_size)
    total_query_samples = get_total_size(query_file_list, h5_key)

    for chunk in tqdm(
        query_iter, total=(total_query_samples // batch_size) + 1, desc="Searching"
    ):
        batch_np = chunk.astype("float32")
        faiss.normalize_L2(batch_np)

        D, I = index.search(batch_np, search_k)

        if use_gpu:
            I = I if isinstance(I, np.ndarray) else I.cpu().numpy()

        if not is_cross_modal:
            cleaned_batch = []
            for i in range(I.shape[0]):
                query_id = global_idx + i
                row = I[i]
                valid_mask = row != query_id
                cleaned_batch.append(row[valid_mask][:k])
            all_indices.append(np.array(cleaned_batch))
        else:
            all_indices.append(I[:, :k])

        global_idx += batch_np.shape[0]

    print("Concatenating ...")
    final_indices = np.concatenate(all_indices, axis=0)
    print(f"Saving to {output_path} ...")
    np.save(output_path, final_indices)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--index", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--h5_key", type=str, default="embeddings")
    parser.add_argument("--fast", action="store_true", help="Use IVFPQ (Fast Mode)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    args = parser.parse_args()

    generate_neighbor_indices(
        query_file_list=args.query,
        output_path=args.output,
        index_file_list=args.index,
        k=args.k,
        batch_size=args.batch_size,
        h5_key=args.h5_key,
        fast_mode=args.fast,
        use_gpu=args.gpu,
    )
