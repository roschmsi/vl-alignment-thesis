from .embedding_data import (
    H5BimodalDataset,
    H5EmbeddingIterableDataset,
    H5UnimodalDataset,
    PairedSubsetDataset,
    UnpairedSubsetDataset,
    VLEmbeddingDataset,
    MMAPDataset,
    build_pairing_plan,
    custom_collate_fn,
)
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from multiprocessing import Value
from torch.utils.data import DataLoader
import numpy as np
import h5py


def get_total_h5_length(paths, key="embeddings"):
    """Helper to count total samples without loading data."""
    total = 0
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
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    data_info: dict = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


@dataclass
class SemiSupervisedDataInfo:
    bimodal_loader: DataLoader
    text_loader: DataLoader
    image_loader: DataLoader
    data_info: dict = None


# def get_embedding_dataset(
#     text_embedding_list,
#     image_embedding_list,
#     extra_text_embedding_list,
#     workers,
#     batch_size,
#     train_num_samples=None,
#     is_train=True,
#     distributed=False,
#     hidden_states=False,
#     hidden_states_img_idx=None,
#     hidden_states_text_idx=None,
#     metadata_path=None,
#     mmap=False,
#     hdf5=False,
# ):
#     assert (
#         text_embedding_list and image_embedding_list
#     ), "Please provide text_embedding_list and image_embedding_list"

#     if mmap:
#         dataset = MMAPDataset(
#             text_embedding_list=text_embedding_list,
#             image_embedding_list=image_embedding_list,
#             extra_text_embedding_list=extra_text_embedding_list,
#             metadata_path=metadata_path,
#             train_num_samples=train_num_samples,
#             hidden_states=hidden_states,
#             hidden_states_img_idx=hidden_states_img_idx,
#             hidden_states_text_idx=hidden_states_text_idx,
#         )
#     elif hdf5:
#         dataset = H5EmbeddingIterableDataset(
#             text_embedding_list=text_embedding_list,
#             image_embedding_list=image_embedding_list,
#             extra_text_embedding_list=extra_text_embedding_list,
#             hidden_states=hidden_states,
#             hidden_states_img_idx=hidden_states_img_idx,
#             hidden_states_text_idx=hidden_states_text_idx,
#         )
#     else:
#         dataset = VLEmbeddingDataset(
#             text_embedding_list,
#             image_embedding_list,
#             extra_text_embedding_list,
#             train_num_samples,
#             hidden_states,
#         )

#     num_samples = len(dataset)
#     sampler = DistributedSampler(dataset) if distributed and is_train else None
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         collate_fn=custom_collate_fn,
#         num_workers=workers,
#         pin_memory=False,
#         prefetch_factor=1,
#         persistent_workers=(workers > 0),
#         sampler=None,
#         drop_last=is_train,
#     )
#     dataloader.num_samples = num_samples
#     dataloader.num_batches = len(dataloader)

#     return DataInfo(
#         dataloader,
#         sampler,
#         data_info={
#             "num_samples": num_samples,
#             "visual_dim": dataset.visual_dim,
#             "text_dim": dataset.text_dim,
#         },
#     )


def get_embedding_dataset(
    text_embedding_list,
    image_embedding_list,
    extra_text_embedding_list,
    workers,
    batch_size,
    train_num_samples=None,
    is_train=True,
    distributed=False,
    hidden_states=False,
    hidden_states_img_idx=None,
    hidden_states_text_idx=None,
    metadata_path=None,
    mmap=False,
    hdf5=False,
    semisupervised=False,
    supervised=False,
    n_supervised_pairs=None,
    batch_size_supervised=None,
    debugging=False,
):
    assert (
        text_embedding_list and image_embedding_list
    ), "Please provide text_embedding_list and image_embedding_list"

    if semisupervised:
        # TODO only for debugging
        if debugging:
            total_samples = 100000
        else:
            total_samples = get_total_h5_length(text_embedding_list, key="embeddings")
            print(f"Total samples found: {total_samples}")

        # split into supervised and unsupervised indices
        all_indices = np.random.permutation(total_samples)
        supervised_indices = all_indices[:n_supervised_pairs]
        unsupervised_indices = all_indices[n_supervised_pairs:]

        print(f"Supervised indices: {len(supervised_indices)}")
        print(f"Unsupervised indices: {len(unsupervised_indices)}")

        # dataset with ground-truth image-text pairs
        bimodal_dataset = H5BimodalDataset(
            text_paths=text_embedding_list,
            image_paths=image_embedding_list,
            indices=supervised_indices,
            h5_key="embeddings",
        )

        # dataset with images only
        image_dataset = H5UnimodalDataset(
            paths=image_embedding_list,
            indices=unsupervised_indices,
            h5_key="embeddings",
        )

        # dataset with text only
        text_dataset = H5UnimodalDataset(
            paths=text_embedding_list,
            indices=unsupervised_indices,
            h5_key="embeddings",
        )

        # since data is in RAM, should we set num_workers=0?
        bimodal_loader = DataLoader(
            bimodal_dataset,
            shuffle=True,
            batch_size=batch_size_supervised,
            num_workers=(workers // 3),
            pin_memory=True,
            drop_last=False,
        )

        image_loader = DataLoader(
            image_dataset,
            shuffle=True,
            batch_size=batch_size - batch_size_supervised,
            num_workers=(workers // 3),
            pin_memory=True,
            drop_last=True,
        )

        text_loader = DataLoader(
            text_dataset,
            shuffle=True,
            batch_size=batch_size - batch_size_supervised,
            num_workers=(workers // 3),
            pin_memory=True,
            drop_last=True,
        )

        bimodal_loader.num_samples = len(bimodal_dataset)
        bimodal_loader.num_batches = len(bimodal_loader)
        image_loader.num_samples = len(image_dataset)
        image_loader.num_batches = len(image_loader)
        text_loader.num_samples = len(text_dataset)
        text_loader.num_batches = len(text_loader)

        return SemiSupervisedDataInfo(
            data_info={
                "num_samples_paired": len(bimodal_dataset),
                "num_samples_unpaired": len(image_dataset),
                "visual_dim": image_dataset[0].shape[0],
                "text_dim": text_dataset[0].shape[0],
            },
            bimodal_loader=bimodal_loader,
            image_loader=image_loader,
            text_loader=text_loader,
        )

    elif supervised:
        # dataset with ground-truth image-text pairs
        total_samples = get_total_h5_length(text_embedding_list, key="embeddings")
        print(f"Total samples found: {total_samples}")

        # split into supervised and unsupervised indices
        all_indices = np.random.permutation(total_samples)

        bimodal_dataset = H5BimodalDataset(
            text_paths=text_embedding_list,
            image_paths=image_embedding_list,
            indices=all_indices,
            h5_key="embeddings",
        )

        # since data is in RAM, should we set num_workers=0?
        bimodal_loader = DataLoader(
            bimodal_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )

        image_loader = None
        text_loader = None

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
            image_loader=image_loader,
            text_loader=text_loader,
        )

    if mmap:
        dataset = MMAPDataset(
            text_embedding_list=text_embedding_list,
            image_embedding_list=image_embedding_list,
            extra_text_embedding_list=extra_text_embedding_list,
            metadata_path=metadata_path,
            train_num_samples=train_num_samples,
            hidden_states=hidden_states,
            hidden_states_img_idx=hidden_states_img_idx,
            hidden_states_text_idx=hidden_states_text_idx,
        )
    elif hdf5:
        # supervised SAIL baseline with limited number of pairs
        if n_supervised_pairs is not None:

            temp_scanner = _H5BaseDataset(text_embedding_list)
            total_samples = temp_scanner.total_samples
            temp_scanner.close()

            print(f"Total samples found: {total_samples}")

            all_indices = np.random.permutation(total_samples)
            supervised_indices = all_indices[:n_supervised_pairs]

            print(f"Supervised indices: {len(supervised_indices)}")

            bimodal_dataset = H5BimodalDataset(
                text_paths=text_embedding_list,
                image_paths=image_embedding_list,
                indices=supervised_indices,
                h5_key="embeddings",
            )

            bimodal_loader = DataLoader(
                bimodal_dataset,
                shuffle=True,
                batch_size=batch_size_supervised,
                num_workers=workers,
                pin_memory=False,
                prefetch_factor=1,
                persistent_workers=(workers > 0),
                drop_last=False,
            )
            bimodal_loader.num_samples = len(bimodal_dataset)
            bimodal_loader.num_batches = len(bimodal_loader)

            return DataInfo(
                dataloader=bimodal_loader,
                sampler=None,
                data_info={
                    "num_samples": bimodal_loader.num_samples,
                    "visual_dim": 2048,  # TODO: do not hardcode
                    "text_dim": 4096,
                },
            )

        else:

            dataset = H5EmbeddingIterableDataset(
                text_embedding_list=text_embedding_list,
                image_embedding_list=image_embedding_list,
                extra_text_embedding_list=extra_text_embedding_list,
                hidden_states=hidden_states,
                hidden_states_img_idx=hidden_states_img_idx,
                hidden_states_text_idx=hidden_states_text_idx,
            )
    else:
        dataset = VLEmbeddingDataset(
            text_embedding_list,
            image_embedding_list,
            extra_text_embedding_list,
            train_num_samples,
            hidden_states,
        )

        # num_samples = len(dataset)
        # K = min(int(n_supervised_pairs), num_samples)

        # paired_idx, unsup_t, unsup_v = build_pairing_plan(
        #     num_samples=num_samples, k_supervised=K, seed=int(seed)
        # )

        # paired_ds = PairedSubsetDataset(dataset, paired_idx)
        # # If K == num_samples, unpaired would be empty; guard it:
        # unpaired_ds = None
        # if num_samples - K > 0:
        #     unpaired_ds = UnpairedSubsetDataset(dataset, unsup_t, unsup_v,
        #                                         reshuffle_each_epoch=is_train, seed=int(seed))

        # # Sampler is None for IterableDatasets that already shard internally
        # paired_loader = DataLoader(
        #     paired_ds,
        #     batch_size=n_supervised_pairs,
        #     collate_fn=custom_collate_fn,
        #     num_workers=workers,
        #     pin_memory=False,
        #     prefetch_factor=1,
        #     persistent_workers=(workers > 0),
        #     drop_last=is_train,
        # )
        # paired_loader.num_samples = len(paired_ds)
        # paired_loader.num_batches = len(paired_loader)

        # if unpaired_ds is not None:
        #     # TODO decouple supervised pairs and batch size
        #     unpaired_loader = DataLoader(
        #         unpaired_ds,
        #         batch_size=batch_size - n_supervised_pairs,
        #         collate_fn=custom_collate_fn,
        #         num_workers=workers,
        #         pin_memory=False,
        #         prefetch_factor=1,
        #         persistent_workers=(workers > 0),
        #         drop_last=is_train,
        #     )
        #     unpaired_loader.num_samples = len(unpaired_ds)
        #     unpaired_loader.num_batches = len(unpaired_loader)
        # else:
        #     unpaired_loader = None

        # # Return BOTH via a tiny struct so caller can grab either stream
        # return SemiSupervisedDataInfo(
        #     data_info={"num_samples_paired": len(paired_ds),
        #                 "num_samples_unpaired": len(unpaired_ds),
        #                 "visual_dim": dataset.visual_dim,
        #                 "text_dim": dataset.text_dim},
        #     paired_loader = paired_loader,
        #     unpaired_loader = unpaired_loader,
        # )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if distributed and is_train else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        num_workers=workers,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=(workers > 0),
        sampler=None,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(
        dataloader,
        sampler,
        data_info={
            "num_samples": num_samples,
            "visual_dim": dataset.visual_dim,
            "text_dim": dataset.text_dim,
        },
    )


def get_data(args, epoch=0):
    data = {}
    if args.text_embedding_list and args.image_embedding_list:
        data["train"] = get_embedding_dataset(
            args.text_embedding_list,
            args.image_embedding_list,
            args.extra_text_embedding_list,
            workers=args.workers,
            batch_size=args.batch_size,
            train_num_samples=args.train_num_samples,
            is_train=True,
            distributed=args.distributed,
            hidden_states=args.hidden_states,
            hidden_states_img_idx=args.hidden_states_img_idx,
            hidden_states_text_idx=args.hidden_states_text_idx,
            metadata_path=args.metadata_path,
            mmap=args.mmap,
            hdf5=args.hdf5,
            semisupervised=args.semisupervised,
            supervised=args.supervised,
            n_supervised_pairs=args.n_supervised_pairs,
            batch_size_supervised=args.batch_size_supervised,
            debugging=args.debugging,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    if args.val_text_embedding_list and args.val_image_embedding_list:
        data["val"] = get_embedding_dataset(
            args.val_text_embedding_list,
            args.val_image_embedding_list,
            extra_text_embedding_list=None,
            workers=args.workers,
            batch_size=args.batch_size,
            train_num_samples=None,
            is_train=False,
            distributed=args.distributed,
            hidden_states=args.hidden_states,
            hidden_states_img_idx=args.hidden_states_img_idx,
            hidden_states_text_idx=args.hidden_states_text_idx,
            metadata_path=args.metadata_path,
            mmap=args.mmap,
            hdf5=args.hdf5,
        )

    return data
