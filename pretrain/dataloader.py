from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import numpy as np


class RandomDataset(Dataset):
    def __init__(self, input, scaler=None, transform_fn=lambda x: x, indices=None, k=5, chunk_size=96):
        self.input, self.indices, self.data = input, indices, None
        n_samples, sample_size = 0, (2 * k + 1) * chunk_size
        with h5py.File(input, 'r') as input:
            if indices is None:
                self.indices = input.keys()
            for idx in self.indices:
                n_samples += len(input[idx]) // sample_size
        
        self.n_samples, self.k, self.chunk_size = n_samples, k, chunk_size
        self.transform_fn = transform_fn
        self.scaler = scaler
            
    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File('self.input', 'r')

        k, chunk_size, sample_size = self.k,\
            self.chunk_size, (2 * self.k + 1) * self.chunk_size
        index = self.indices[np.random.randint(len(self.indices))]
        end_pos = len(self.data[index]) - sample_size
        if end_pos < 0:
            sample = self.data[index][()]
        else:
            init_pos = np.random.randint(end_pos)
            sample = self.data[index][init_pos: init_pos + sample_size]
        
        sample = sample if self.scaler is not None\
            else self.scaler.transform(sample)
        
        sample = sample.reshape(k * 2 + 1, chunk_size, -1)
        targets, feats = sample[k], np.concatenate(
            [sample[:k,:,:], sample[k+1:,:,:]], axis=0).reshape(2 * k * chunk_size, -1)
        feats, targets = torch.from_numpy(feats), torch.from_numpy(targets)

        return self.transform_fn(feats), targets

    def __len__(self):
        return self.n_samples * 2


def create_dataloader(input, indices=None, is_random=True, scaler=None, **kwargs):
    kwargs.setdefault('dataloader_args', 
        {'batch_size': 2048, 'num_workers': 4, 'shuffle': True})
    kwargs.setdefault('sample_args', {'k': 5, 'chunk_size': 96})

    if is_random:
        _dataset = RandomDataset(input, indices=indices, scaler=scaler, **kwargs['sample_args'])

    return DataLoader(_dataset, **kwargs['dataloader_args'])
