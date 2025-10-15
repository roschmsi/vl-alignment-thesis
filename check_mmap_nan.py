import numpy as np
from math import ceil
from tqdm.auto import tqdm

path = "/dss/mcmlscratch/07/ga27tus3/mmap_data/NV-Embed-v2/dreamclipcc12m_shortSV_captions.mmap"

# adjust to your file
dtype = np.float16
shape = (10012845, 4096, 11)

# use r+ to allow in-place edits
mm = np.memmap(path, mode="r+", dtype=dtype, shape=shape)

if not np.issubdtype(mm.dtype, np.floating):
    print("Non-floating dtype → cannot contain ±inf.")
else:
    flat = mm.reshape(-1)  # view; in-place edits propagate to 'mm'
    N = flat.size
    chunk_elems = 5_000_000  # lower if you need less RAM

    tmp = np.empty(min(chunk_elems, N), dtype=bool)  # reusable mask buffer
    found_inf = False
    capped_count = 0

    num_chunks = ceil(N / chunk_elems)
    for i in tqdm(range(80000, num_chunks), desc="Scanning & capping ±inf", unit="chunk"):
        start = i * chunk_elems
        end = min(start + chunk_elems, N)
        block = flat[start:end]                  # view, no full copy
        mv = tmp[: end - start]                  # mask view sized to chunk

        # mark ±inf in this chunk
        np.isinf(block, out=mv)

        if mv.any():
            found_inf = True
            print(f'start: {start}')
            print(f'end: {end}')

            # cap +inf → +100, -inf → -100 (avoid creating large temporaries)
            pos = mv & (block > 0)
            neg = mv & (block < 0)
            block[pos] = np.float16(100)
            block[neg] = np.float16(-100)

            capped_count += int(mv.sum())

    # persist updates to disk
    mm.flush()

    print("Found ±inf:", found_inf, "| Values capped:", capped_count)
