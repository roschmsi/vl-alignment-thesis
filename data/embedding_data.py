import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import glob
from natsort import natsorted
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pdb

def custom_collate_fn(batch):
    if len(batch[0]) == 3:
        text_vectors, image_vectors, extra_text_vectors = zip(*batch)
        return torch.stack(text_vectors, 0), torch.stack(image_vectors, 0), torch.stack(extra_text_vectors, 0)
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
        hidden_states_idx=[idx1, idx2, -1]

    features = features[:, hidden_states_idx]
    features = features.flatten()

    return features

def load_vectors(embedding_dirs: list[str], hidden_states: bool = False,
                      dtype: torch.dtype = torch.float16) -> torch.Tensor:
    files = []
    for d in embedding_dirs:
        files.extend(natsorted(glob.glob(os.path.join(d, "*.pt"))))

    # TODO remove for full dataset
    files = files[:2000]

    chunks = []
    with torch.no_grad():
        for file in tqdm(files, desc="Loading embedding data", unit="file"):
            x = torch.load(file, map_location="cpu", weights_only=True)  # [B, L, D] or [B, D]
            # x = extract_hidden_states(x) if hidden_states else x[..., -1]
            if x.dtype is not dtype:
                x = x.to(dtype)
            # x = x.contiguous()
            x = x.clone()
            chunks.append(x)
    return torch.cat(chunks, dim=0)

class VLEmbeddingDataset(Dataset):
    def __init__(self, text_embedding_list, image_embedding_list, extra_text_embedding_list=None, train_num_samples=None, hidden_states=False):

        self.text_vectors, self.image_vectors = self._load_image_text_vectors(image_embedding_list, text_embedding_list, hidden_states)
        n_img, n_txt = len(self.image_vectors), len(self.text_vectors)
        assert n_img > 0 and n_txt > 0 and n_txt % n_img == 0, f"text vectors length ({n_txt}) is not a multiple of image vectors length ({n_img})"

        if extra_text_embedding_list:
            print(f"Loading extra text vectors from {extra_text_embedding_list}")
            self.extra_text_vectors, _ = self._load_image_text_vectors(text_embedding_list = extra_text_embedding_list, hidden_states=hidden_states)
            assert len(self.extra_text_vectors) == len(self.text_vectors), f"extra text vectors length {len(self.extra_text_vectors)} is not equal to text vectors length {len(self.text_vectors)}"
    
        if train_num_samples is not None:
            num_samples = len(self.text_vectors)
            random_indices = np.random.choice(num_samples, train_num_samples, replace=False)
            self.text_vectors = [self.text_vectors[i] for i in random_indices]
            self.image_vectors = [self.image_vectors[i] for i in random_indices]
            if extra_text_embedding_list:
                self.extra_text_vectors = [self.extra_text_vectors[i] for i in random_indices]
            print(f"Random Selecting {train_num_samples} samples as training data")

        self.image_num = len(self.image_vectors)
        self.text_num = len(self.text_vectors)

        self.visual_dim = self.image_vectors[0].shape[0]
        self.text_dim = self.text_vectors[0].shape[0]
        
    def _load_image_text_vectors(self, image_embedding_list = None, text_embedding_list = None, hidden_states=False):
        assert image_embedding_list is not None or text_embedding_list is not None, "Either image_embedding_list or text_embedding_list must be provided"
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
        
        if hasattr(self, 'extra_text_vectors'):
            return self.text_vectors[idx], self.image_vectors[img_idx], self.extra_text_vectors[idx]
        else:
            return self.text_vectors[idx], self.image_vectors[img_idx]


class MMAPDataset(Dataset):
    def __init__(self, text_embedding_list, image_embedding_list, extra_text_embedding_list, metadata_path=None, train_num_samples=None, hidden_states=False, hidden_states_img_idx=None, hidden_states_text_idx=None):
        super().__init__()

        metadata = torch.load(metadata_path)
        self.num_samples = metadata['num_samples']
        vision_shape = tuple(metadata['vision_shape'])
        text_shape = tuple(metadata['text_shape'])
        vision_dtype = np.dtype(metadata['vision_dtype'])
        text_dtype = np.dtype(metadata['text_dtype'])

        self.hidden_states = hidden_states
        self.hidden_states_text_idx = hidden_states_text_idx
        self.hidden_states_img_idx = hidden_states_img_idx

        # TODO adapt dealing with list when CC3M, CC12M, YFCC15M are merged
        self.image_vectors = np.memmap(
            image_embedding_list[0], 
            dtype=vision_dtype, 
            mode='r',
            shape=(self.num_samples, *vision_shape)
        )
        self.text_vectors = np.memmap(
            text_embedding_list[0], 
            dtype=text_dtype, 
            mode='r',
            shape=(self.num_samples, *text_shape)
        )

        n_img, n_txt = len(self.image_vectors), len(self.text_vectors)
        assert n_img > 0 and n_txt > 0 and n_txt % n_img == 0, f"text vectors length ({n_txt}) is not a multiple of image vectors length ({n_img})"

        if extra_text_embedding_list:
            self.extra_text_vectors = np.memmap(
                extra_text_embedding_list[0], 
                dtype=text_dtype, 
                mode='r',
                shape=(self.num_samples, *text_shape)
            )
            
            assert len(self.extra_text_vectors) == len(self.text_vectors), f"extra text vectors length {len(self.extra_text_vectors)} is not equal to text vectors length {len(self.text_vectors)}"

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

        if hasattr(self, 'extra_text_vectors'):
            extra = np.ascontiguousarray(self.extra_text_vectors[idx])
            if self.hidden_states:
                extra = extract_hidden_states(extra, self.hidden_states_text_idx)
            else:
                extra = extra[:, -1]

            return torch.from_numpy(txt), torch.from_numpy(img), torch.from_numpy(extra)
        
        return torch.from_numpy(txt), torch.from_numpy(img)


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
    def __init__(self, text_files, image_files, extra_text_files=None, hidden_states=False):
        self.text_files = text_files
        self.image_files = image_files
        self.extra_text_files = extra_text_files
        self.hidden_states = hidden_states

        assert len(self.text_files) > 0, "No text embedding files found."
        assert len(self.text_files) == len(self.image_files), "Mismatch in number of text and image files"
        if self.extra_text_files:
            assert len(self.extra_text_files) == len(self.text_files), "Mismatch in number of extra text and text files"

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
            extra_text_batch = torch.load(extra_text_file_path, weights_only=True).to(torch.float16)
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
        text_batch_sample = torch.load(self.text_files[0], map_location='cpu', weights_only=True)
        if self.hidden_states:
            text_batch_sample = extract_hidden_states(text_batch_sample)
        else:
            text_batch_sample = text_batch_sample[:, :, -1]
        text_dim = text_batch_sample.shape[-1]
        
        image_batch_sample = torch.load(self.image_files[0], map_location='cpu', weights_only=True)
        if self.hidden_states:
            image_batch_sample = extract_hidden_states(image_batch_sample)
        else:
            image_batch_sample = image_batch_sample[:, :, -1]
        visual_dim = image_batch_sample.shape[-1]
        
        return visual_dim, text_dim


if __name__ == "__main__":

    text_embedding_dir = ['/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/text_embedding/gte-large-en-v1.5/validation']
    image_embedding_dir = ['/home/mila/l/le.zhang/scratch/light_align/data/tensor_data/image_embedding/dinov2-large/validation']
    print("Loading dataset...")
    # dataset = LazyVLEmbeddingDataset(text_embedding_dir, image_embedding_dir)
    dataset = VLEmbeddingDataset(text_embedding_dir, image_embedding_dir)
    print("Dataset loaded.")
    # 创建DataLoader
    breakpoint()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    for batch in tqdm(dataloader):
        if len(batch) == 3:
            text_vectors, image_vectors, extra_text_vectors = batch
        else:
            text_vectors, image_vectors = batch
            extra_text_vectors = None
        print(text_vectors.shape, image_vectors.shape)

