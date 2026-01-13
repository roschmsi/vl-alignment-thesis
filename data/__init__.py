from .embedding_data import (
    H5BimodalDataset,
    H5BimodalDatasetWithNNPositives,
    H5UnimodalDataset,
)
from dataclasses import dataclass
from multiprocessing import Value
from torch.utils.data import DataLoader
import numpy as np
import h5py


def get_total_h5_length(paths, key="embeddings", multi_text_mode=False):
    if not paths:
        return 0

    if not multi_text_mode:
        total = 0
        for p in paths:
            with h5py.File(p, "r") as f:
                total += len(f[key])
        return total

    else:
        ref_len = None
        for p in paths:
            with h5py.File(p, "r") as f:
                curr_len = len(f[key])

            if ref_len is None:
                ref_len = curr_len
            elif curr_len != ref_len:
                raise ValueError(
                    f"Size mismatch: {paths[0]} has {ref_len}, but {p} has {curr_len}."
                )
        return ref_len


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class SemiSupervisedDataInfo:
    bimodal_loader: DataLoader
    text_loader: DataLoader
    image_loader: DataLoader
    data_info: dict = None


@dataclass
class DataInfo:
    dataloader: DataLoader


def get_embedding_dataset(
    supervised_text_embedding,
    supervised_image_embedding,
    unsupervised_text_embedding,
    unsupervised_image_embedding,
    workers,
    batch_size,
    semisupervised=False,
    supervised=False,
    n_supervised_pairs=None,
    n_unsupervised_text=None,
    n_unsupervised_image=None,
    batch_size_supervised=None,
    unsupervised_index_mode="random",  # "aligned", "disjoint", "random"
    debugging=False,
    multi_text_mode=False,
    # NNCLR parameters
    text_nn_positives=0,
    image_nn_positives=0,
    text_neighbors_path=None,
    image_neighbors_path=None,
    text_topk=0,
    image_topk=0,
):
    # Assert supervised text and image files are parallel and have same length
    len_text = get_total_h5_length(
        supervised_text_embedding, key="embeddings", multi_text_mode=multi_text_mode
    )
    len_image = get_total_h5_length(supervised_image_embedding, key="embeddings")

    assert len_text == len_image, (
        f"ERROR: Supervised datasets must have the same length\n"
        f"Text file: {len_text} samples\n"
        f"Image file: {len_image} samples\n"
    )

    total_sup_samples = len_text

    # Reduce set of indices for debugging
    if debugging:
        n_supervised_pairs = 10000
        batch_size_supervised = 10000
        n_unsupervised_image = 100000
        n_unsupervised_text = 100000
        total_sup_samples = 110000

    # Generate master permutation for the supervised files
    # Ensure text[i] and image[i] remain aligned
    sup_permutation = np.random.permutation(total_sup_samples)

    # Determine cut-off for supervised pairs
    if n_supervised_pairs is not None:
        # Validate the specific user input
        assert n_supervised_pairs <= total_sup_samples, (
            f"Requested more supervised pairs ({n_supervised_pairs}) "
            f"than available ({total_sup_samples})."
        )
        n_sup = n_supervised_pairs
    else:
        # Default use everything
        n_sup = total_sup_samples

    sup_indices = sup_permutation[:n_sup]

    # The remaining indices in the supervised file (may be used as unsupervised samples)
    remaining_sup_indices = sup_permutation[n_sup:]

    # Check per modality if unsupervised source is shared or distinct
    is_shared_text = unsupervised_text_embedding == supervised_text_embedding
    is_shared_image = unsupervised_image_embedding == supervised_image_embedding

    # Create pool of text indices
    if is_shared_text:
        # Take from the remaining supervised indices
        len_ut_source = len(remaining_sup_indices)
        pool_text = remaining_sup_indices
    else:
        # Take from a new file
        len_ut_source = get_total_h5_length(
            unsupervised_text_embedding, key="embeddings"
        )
        pool_text = np.random.permutation(len_ut_source)

    # Create pool of image indices
    if is_shared_image:
        # Take from the remaining supervised indices
        len_ui_source = len(remaining_sup_indices)
        pool_image = remaining_sup_indices
    else:
        # Take from a new file
        len_ui_source = get_total_h5_length(
            unsupervised_image_embedding, key="embeddings"
        )
        pool_image = np.random.permutation(len_ui_source)

    # Determine if unsupervised sources are compatible (same dataset)
    unsupervised_are_compatible = False

    if is_shared_text and is_shared_image:
        unsupervised_are_compatible = True
    elif (not is_shared_text) and (not is_shared_image):
        # Extract base name to check if they are the same dataset (e.g. "cc12m_concat.h5" -> "cc12m")
        name_ut = str(unsupervised_text_embedding).split("/")[-1].split("_")[0]
        name_ui = str(unsupervised_image_embedding).split("/")[-1].split("_")[0]
        if name_ut == name_ui and len_ut_source == len_ui_source:
            unsupervised_are_compatible = True

    print(
        f"Unsupervised Sources: Text={'Shared' if is_shared_text else 'Distinct'}, Image={'Shared' if is_shared_image else 'Distinct'}"
    )
    print(f"Compatibility of unsupervised sources: {unsupervised_are_compatible}")

    if unsupervised_are_compatible:
        print(f"Applying mode: {unsupervised_index_mode}")

        if unsupervised_index_mode == "aligned":
            # Force indices to match exactly
            pool_image = pool_text

        elif unsupervised_index_mode == "disjoint":
            # Split pool of indices into two non-overlapping halves
            limit = len(pool_text) // 2
            pool_text = pool_text[:limit]
            pool_image = pool_image[limit:]

        elif unsupervised_index_mode == "random":
            # Randomly permute one of the two modalities
            pool_image = np.random.permutation(pool_image)

    else:
        print(
            "Unsupervised text and image sources are different. Indices are selected independently."
        )
        pass

    # final selection of unsupervised indices based on number
    n_ut = n_unsupervised_text if n_unsupervised_text is not None else len(pool_text)
    n_ut = min(n_ut, len(pool_text))
    unsup_text_indices = pool_text[:n_ut]

    n_ui = n_unsupervised_image if n_unsupervised_image is not None else len(pool_image)
    n_ui = min(n_ui, len(pool_image))

    # resolve case where n_unsupervised_text != n_unsupervised_image but aligned indices are required
    if unsupervised_index_mode == "aligned" and unsupervised_are_compatible:
        limit = min(n_ut, n_ui)
        unsup_text_indices = pool_text[:limit]
        unsup_image_indices = pool_image[:limit]
    else:
        unsup_image_indices = pool_image[:n_ui]

    print(
        f"Final Unsupervised Counts: {len(unsup_text_indices)} Text, {len(unsup_image_indices)} Image"
    )

    if semisupervised:
        # Supervised image-text dataset (paired)
        bimodal_dataset = H5BimodalDataset(
            text_paths=supervised_text_embedding,
            image_paths=supervised_image_embedding,
            indices=sup_indices,
            h5_key="embeddings",
            multi_text_mode=multi_text_mode,
        )

        bimodal_loader = DataLoader(
            bimodal_dataset,
            shuffle=True,
            batch_size=batch_size_supervised,
            num_workers=0,  # (workers // 3),
            pin_memory=True,
            drop_last=False,
        )

        # Unsupervised image dataset
        image_loader = None
        vis_dim = bimodal_dataset[0][1].shape[0]
        if len(unsup_image_indices) > 0:
            image_dataset = H5UnimodalDataset(
                paths=unsupervised_image_embedding,
                indices=unsup_image_indices,
                h5_key="embeddings",
            )
            image_loader = DataLoader(
                image_dataset,
                shuffle=True,
                batch_size=min(
                    batch_size - batch_size_supervised, n_unsupervised_image
                ),
                num_workers=0,  # (workers // 3),
                pin_memory=True,
                drop_last=True,
            )
            image_loader.num_samples = len(image_dataset)
            image_loader.num_batches = len(image_loader)
            if len(image_dataset) > 0:
                vis_dim = image_dataset[0].shape[0]

        # Unsupervised text dataset
        text_loader = None
        txt_dim = bimodal_dataset[0][0].shape[0]
        if len(unsup_text_indices) > 0:
            text_dataset = H5UnimodalDataset(
                paths=unsupervised_text_embedding,
                indices=unsup_text_indices,
                h5_key="embeddings",
            )
            text_loader = DataLoader(
                text_dataset,
                shuffle=True,
                batch_size=min(batch_size - batch_size_supervised, n_unsupervised_text),
                num_workers=0,  # (workers // 3),
                pin_memory=True,
                drop_last=True,
            )
            text_loader.num_samples = len(text_dataset)
            text_loader.num_batches = len(text_loader)
            if len(text_dataset) > 0:
                txt_dim = text_dataset[0].shape[0]

        bimodal_loader.num_samples = len(bimodal_dataset)
        bimodal_loader.num_batches = len(bimodal_loader)

        return SemiSupervisedDataInfo(
            data_info={
                "num_samples_paired": len(bimodal_dataset),
                "num_samples_unpaired_image": len(unsup_image_indices),
                "num_samples_unpaired_text": len(unsup_text_indices),
                "visual_dim": vis_dim,
                "text_dim": txt_dim,
            },
            bimodal_loader=bimodal_loader,
            image_loader=image_loader,
            text_loader=text_loader,
        )

    elif supervised:
        if image_nn_positives > 0 or text_nn_positives > 0:
            bimodal_dataset = H5BimodalDatasetWithNNPositives(
                text_paths=supervised_text_embedding,
                image_paths=supervised_image_embedding,
                sup_indices=sup_indices,
                text_neighbors_path=text_neighbors_path,
                image_neighbors_path=image_neighbors_path,
                text_nn_positives=text_nn_positives,
                image_nn_positives=image_nn_positives,
                text_topk=text_topk,
                image_topk=image_topk,
                h5_key="embeddings",
            )
        else:
            bimodal_dataset = H5BimodalDataset(
                text_paths=supervised_text_embedding,
                image_paths=supervised_image_embedding,
                indices=sup_indices,
                h5_key="embeddings",
                multi_text_mode=multi_text_mode,
            )

        bimodal_loader = DataLoader(
            bimodal_dataset,
            shuffle=True,
            batch_size=min(batch_size, n_supervised_pairs),
            num_workers=0,  # workers,
            pin_memory=True,
            drop_last=True,
        )

        bimodal_loader.num_samples = len(bimodal_dataset)
        bimodal_loader.num_batches = len(bimodal_loader)

        vis_dim = bimodal_dataset[0][1].shape[0]
        txt_dim = bimodal_dataset[0][0].shape[0]

        return SemiSupervisedDataInfo(
            data_info={
                "num_samples_paired": len(bimodal_dataset),
                "num_samples_unpaired": 0,
                "visual_dim": vis_dim,
                "text_dim": txt_dim,
            },
            bimodal_loader=bimodal_loader,
            image_loader=None,
            text_loader=None,
        )
    else:
        raise ValueError("Must specify either supervised or semisupervised mode.")


def make_bimodal_validation_loader(
    text_paths,
    image_paths,
    batch_size=2048,
    num_workers=8,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    h5_key="embeddings",
):
    # normalize to list
    if isinstance(text_paths, str):
        text_paths = [text_paths]
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # total length (must match)
    def total_len(paths):
        n = 0
        for p in paths:
            with h5py.File(p, "r") as f:
                n += f[h5_key].shape[0]
        return n

    nt = total_len(text_paths)
    ni = total_len(image_paths)
    if nt != ni:
        raise ValueError(f"Length mismatch: text={nt}, image={ni}")

    indices = np.arange(nt, dtype=np.int64)

    ds = H5BimodalDataset(
        text_paths=text_paths,
        image_paths=image_paths,
        indices=indices,
        h5_key=h5_key,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    loader.num_samples = len(ds)
    loader.num_batches = len(loader)

    return DataInfo(dataloader=loader)


def get_data(args, epoch=0):
    data = {}

    if args.supervised_text_embedding and args.supervised_image_embedding:
        data["train"] = get_embedding_dataset(
            supervised_text_embedding=args.supervised_text_embedding,
            supervised_image_embedding=args.supervised_image_embedding,
            unsupervised_text_embedding=args.unsupervised_text_embedding,
            unsupervised_image_embedding=args.unsupervised_image_embedding,
            workers=args.workers,
            batch_size=args.batch_size,
            semisupervised=args.semisupervised,
            supervised=args.supervised,
            n_supervised_pairs=args.n_supervised_pairs,
            n_unsupervised_image=args.n_unsupervised_image,
            n_unsupervised_text=args.n_unsupervised_text,
            batch_size_supervised=args.batch_size_supervised,
            unsupervised_index_mode=args.unsupervised_index_mode,
            multi_text_mode=args.multi_text_mode,
            debugging=args.debugging,
            text_nn_positives=args.text_nn_positives,
            image_nn_positives=args.image_nn_positives,
            text_neighbors_path=args.text_neighbors_path,
            image_neighbors_path=args.image_neighbors_path,
            text_topk=args.text_topk,
            image_topk=args.image_topk,
        )
        data["val"] = make_bimodal_validation_loader(
            text_paths=args.val_text_embedding,
            image_paths=args.val_image_embedding,
            batch_size=2048,
            num_workers=args.workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            h5_key="embeddings",
        )
    else:
        raise ValueError("Supervised text/image embedding paths are required.")

    # TODO logic for validation data

    return data
