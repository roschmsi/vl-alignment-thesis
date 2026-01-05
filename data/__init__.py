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


def get_total_h5_length(paths, key="embeddings"):
    """Helper to count total samples without loading data."""
    total = 0
    if not paths:
        return 0
    for p in paths:
        with h5py.File(p, "r") as f:
            total += len(f[key])
    return total


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
    debugging=False,
    text_nn_positives=0,
    image_nn_positives=0,
    text_neighbors_path=None,
    image_neighbors_path=None,
    text_topk=0,
    image_topk=0,
):
    # Determine supervised indices (anchors)
    if debugging:
        total_sup_samples = 100000
    else:
        # Assumes supervised text and image files are parallel and have same length
        total_sup_samples = get_total_h5_length(
            supervised_text_embedding, key="embeddings"
        )

    # Generate a master permutation for the supervised files
    # This ensures text[i] and image[i] remain aligned
    sup_permutation = np.random.permutation(total_sup_samples)

    # Determine cut-off for supervised pairs
    n_sup = n_supervised_pairs if n_supervised_pairs is not None else total_sup_samples
    n_sup = min(n_sup, total_sup_samples)

    sup_indices = sup_permutation[:n_sup]

    # The remaining indices in the supervised file (available if we need to split)
    remaining_sup_indices = sup_permutation[n_sup:]

    # Determined unsupervised text indices
    unsup_text_indices = []

    # Check if we should split the supervised file or load a new one
    if unsupervised_text_embedding == supervised_text_embedding:
        print("Text Source: Shared (Splitting supervised file)")
        # Use the remainder of the supervised permutation
        n_ut = (
            n_unsupervised_text
            if n_unsupervised_text is not None
            else len(remaining_sup_indices)
        )
        n_ut = min(n_ut, len(remaining_sup_indices))
        unsup_text_indices = remaining_sup_indices[:n_ut]

    elif unsupervised_text_embedding:
        print("Text Source: Distinct (Loading separate file)")
        if debugging:
            total_ut = 100000
        else:
            total_ut = get_total_h5_length(
                unsupervised_text_embedding, key="embeddings"
            )

        # New independent permutation
        perm_ut = np.random.permutation(total_ut)
        n_ut = n_unsupervised_text if n_unsupervised_text is not None else total_ut
        n_ut = min(n_ut, total_ut)
        unsup_text_indices = perm_ut[:n_ut]

    # Determine unsupervised image indices
    unsup_image_indices = []

    # Check if we should split the supervised file or load a new one
    if unsupervised_image_embedding == supervised_image_embedding:
        print("Image Source: Shared (Splitting supervised file)")
        # Use the remainder of the supervised permutation
        # Note: We use the same 'remaining_sup_indices' pool.
        # Ideally, unsup text and unsup image from the *same* split file are not paired,
        # so picking the same indices from the remainder is fine (they are just unpaired samples).
        n_ui = (
            n_unsupervised_image
            if n_unsupervised_image is not None
            else len(remaining_sup_indices)
        )
        n_ui = min(n_ui, len(remaining_sup_indices))
        unsup_image_indices = remaining_sup_indices[:n_ui]
    elif unsupervised_image_embedding:
        print("Image Source: Distinct (Loading separate file)")
        if debugging:
            total_ui = 100000
        else:
            total_ui = get_total_h5_length(
                unsupervised_image_embedding, key="embeddings"
            )

        # New independent permutation
        perm_ui = np.random.permutation(total_ui)
        n_ui = n_unsupervised_image if n_unsupervised_image is not None else total_ui
        n_ui = min(n_ui, total_ui)
        unsup_image_indices = perm_ui[:n_ui]

    print(f"Total Supervised Pairs: {len(sup_indices)}")
    print(f"Total Unsupervised Texts: {len(unsup_text_indices)}")
    print(f"Total Unsupervised Images: {len(unsup_image_indices)}")

    if semisupervised:
        # Supervised image-text dataset (paired)
        bimodal_dataset = H5BimodalDataset(
            text_paths=supervised_text_embedding,
            image_paths=supervised_image_embedding,
            indices=sup_indices,
            h5_key="embeddings",
        )

        bimodal_loader = DataLoader(
            bimodal_dataset,
            shuffle=True,
            batch_size=batch_size_supervised,
            num_workers=(workers // 3),
            pin_memory=True,
            drop_last=False,
        )

        # Unsupervised image dataset
        image_loader = None
        vis_dim = 2048
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
                num_workers=(workers // 3),
                pin_memory=True,
                drop_last=True,
            )
            image_loader.num_samples = len(image_dataset)
            image_loader.num_batches = len(image_loader)
            if len(image_dataset) > 0:
                vis_dim = image_dataset[0].shape[0]

        # Unsupervised text dataset
        text_loader = None
        txt_dim = 4096
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
                num_workers=(workers // 3),
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
            )

        bimodal_loader = DataLoader(
            bimodal_dataset,
            shuffle=True,
            batch_size=min(batch_size, n_supervised_pairs),
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )

        bimodal_loader.num_samples = len(bimodal_dataset)
        bimodal_loader.num_batches = len(bimodal_loader)

        return SemiSupervisedDataInfo(
            data_info={
                "num_samples_paired": len(bimodal_dataset),
                "num_samples_unpaired": 0,
                "visual_dim": 2048,
                "text_dim": 4096,
            },
            bimodal_loader=bimodal_loader,
            image_loader=None,
            text_loader=None,
        )
    else:
        raise ValueError("Must specify either supervised or semisupervised mode.")


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
            debugging=args.debugging,
            text_nn_positives=args.text_nn_positives,
            image_nn_positives=args.image_nn_positives,
            text_neighbors_path=args.text_neighbors_path,
            image_neighbors_path=args.image_neighbors_path,
            text_topk=args.text_topk,
            image_topk=args.image_topk,
        )
    else:
        raise ValueError("Supervised text/image embedding paths are required.")

    # TODO logic for validation data

    return data
