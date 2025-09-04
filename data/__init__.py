from .embedding_data import VLEmbeddingDataset, custom_collate_fn, BatchedLazyDataset, batched_collate_fn
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from multiprocessing import Value
from torch.utils.data import DataLoader
import pdb
from natsort import natsorted
import glob
import os

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

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

def get_embedding_dataset(
        text_embedding_list,
        image_embedding_list,
        extra_text_embedding_list,
        workers,
        batch_size,
        train_num_samples = None,
        is_train = True,
        distributed=False,
        hidden_states=False,
    ):
    assert text_embedding_list and image_embedding_list, "Please provide text_embedding_list and image_embedding_list"
    dataset = VLEmbeddingDataset(
        text_embedding_list,
        image_embedding_list,
        extra_text_embedding_list,
        train_num_samples,
        hidden_states,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)


    return DataInfo(dataloader, sampler, data_info={'num_samples': num_samples, 'visual_dim': dataset.visual_dim, 'text_dim': dataset.text_dim})


# def get_embedding_dataset(
#         text_embedding_list,
#         image_embedding_list,
#         extra_text_embedding_list,
#         workers,
#         batch_size,  # BEHAVIOR CHANGE: This is now interpreted as files_per_batch
#         train_num_samples = None, # BEHAVIOR CHANGE: This is now interpreted as train_num_files
#         is_train = True,
#         distributed=False,
#         hidden_states=False,
#     ):
#     """
#     Creates a DataInfo object with a memory- and I/O-efficient DataLoader.

#     NOTE: The behavior of 'batch_size' and 'train_num_samples' has changed.
#     - `batch_size`: Now specifies the number of .pt FILES to load per batch.
#     - `train_num_samples`: Now specifies the number of file PAIRS to use for training.
#     """
#     assert text_embedding_list and image_embedding_list, "Please provide text_embedding_list and image_embedding_list"

#     def _get_file_paths(embedding_list: list[str]) -> list[str]:
#         if not embedding_list:
#             return []
#         files = []
#         for dir_path in embedding_list:
#             files.extend(natsorted(glob.glob(os.path.join(dir_path, "*.pt"))))
#         return files

#     text_files = _get_file_paths(text_embedding_list)
#     image_files = _get_file_paths(image_embedding_list)
#     extra_text_files = _get_file_paths(extra_text_embedding_list)

#     # Adapt train_num_samples to mean train_num_files
#     if is_train and train_num_samples is not None and train_num_samples < len(text_files):
#         print(f"Randomly selecting {train_num_samples} file pairs for training.")
#         indices = np.random.choice(len(text_files), train_num_samples, replace=False)
#         text_files = [text_files[i] for i in indices]
#         image_files = [image_files[i] for i in indices]
#         if extra_text_files:
#             extra_text_files = [extra_text_files[i] for i in indices]

#     # TODO reset debugging mode
#     dataset = BatchedLazyDataset(
#         text_files, # [:4000],
#         image_files, # [:4000],
#         extra_text_files,
#         hidden_states,
#     )

#     num_samples = dataset.get_total_samples()
#     visual_dim, text_dim = dataset.get_dimensions()
    
#     sampler = DistributedSampler(dataset) if distributed and is_train else None
#     shuffle = is_train and sampler is None

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size, # This is now files_per_batch
#         collate_fn=batched_collate_fn, # Use the new collate function
#         shuffle=shuffle,
#         num_workers=workers,
#         pin_memory=True,
#         sampler=sampler,
#         drop_last=is_train,
#     )
#     dataloader.num_samples = num_samples
#     dataloader.num_batches = len(dataloader)
    
#     data_info_dict = {
#         'num_samples': num_samples,
#         'visual_dim': visual_dim,
#         'text_dim': text_dim
#     }

#     return DataInfo(dataloader, sampler, data_info=data_info_dict)
    

def get_data(args, epoch=0):
    data = {}
    if args.text_embedding_list and args.image_embedding_list:
        data['train'] = get_embedding_dataset(
            args.text_embedding_list,
            args.image_embedding_list,
            args.extra_text_embedding_list,
            workers=args.workers,
            batch_size=args.batch_size,
            train_num_samples=args.train_num_samples,
            is_train=True,
            distributed=args.distributed,
            hidden_states=args.hidden_states,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    if args.val_text_embedding_list and args.val_image_embedding_list:
        data['val'] = get_embedding_dataset(
            args.val_text_embedding_list,
            args.val_image_embedding_list,
            extra_text_embedding_list = None,
            workers=args.workers,
            batch_size=args.batch_size,
            train_num_samples = None,
            is_train=False,
            distributed=args.distributed,
            hidden_states=args.hidden_states,
        )

    return data


