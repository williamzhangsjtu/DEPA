from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import numpy as np


class RandomDataset(Dataset):
    def __init__(self, input, scaler=None, transform_fn=lambda x: x, indices=None, **kwargs):
        chunk_size, k = kwargs.get('chunk_size', 96), kwargs.get('k', 5)
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
            self.data = h5py.File(self.input, 'r')

        k, chunk_size, sample_size = self.k,\
            self.chunk_size, (2 * self.k + 1) * self.chunk_size
        index = self.indices[torch.randint(0, len(self.indices), (1, ))[0]]
        end_pos = len(self.data[index]) - sample_size
        if end_pos < 0:
            sample = self.data[index][()]
            tmp = np.zeros((sample_size, sample.shape[-1]))
            tmp[:len(sample), :] = sample
            sample = tmp
        else:
            init_pos = torch.randint(0, end_pos, (1, ))[0]
            sample = self.data[index][init_pos: init_pos + sample_size]
        sample = sample if self.scaler is None\
            else self.scaler.transform(sample)
        
        sample = sample.reshape(k * 2 + 1, chunk_size, -1)
        targets, feats = sample[k], np.concatenate(
            [sample[:k,:,:], sample[k+1:,:,:]], axis=0).reshape(2 * k * chunk_size, -1)

        return self.transform_fn(torch.from_numpy(feats)).to(torch.float),\
            torch.from_numpy(targets).to(torch.float)

    def __len__(self):
        return self.n_samples * self.k

class DEPADataset(Dataset):
    def __init__(self, input, indices=None, scaler=None, transform_fn=lambda x: x, **kwargs):
        chunk_size, k, alpha = kwargs.get('chunk_size', 96),\
            kwargs.get('k', 5), kwargs.get('alpha', 1.2)

        self.input, self.indices, self.data = input, indices, None
        sample_size = int((2 * k + 1) * chunk_size * alpha)

        self.audio_id, self.start_from = [], {}
        with h5py.File(input, 'r') as input:
            if indices is None:
                self.indices = input.keys()
            init = 0
            for index in indices:
                n_samples = input[index].shape[0] // sample_size
                if not n_samples: continue
                self.start_from[index] = init
                init += n_samples
                self.audio_id += [str(index)] * n_samples
        
        self.sample_size, self.k, self.chunk_size = sample_size, k, chunk_size
        self.transform_fn = transform_fn
        self.scaler = scaler
    
    def __getitem__(self, index):
        chunk_size, k = self.chunk_size, self.k
        if (self.data is None):
            self.data = h5py.File(self.input, 'r')
        audio_id = self.audio_id[index]
        seq_sample = index - self.start_from[str(audio_id)]
        sample = self.data[str(audio_id)][
            self.sample_size * seq_sample: self.sample_size * (seq_sample + 1)]

        sample = sample if self.scaler is None\
            else self.scaler.transform(sample)
        
        sample = sample[:(2 * k + 1) * chunk_size].reshape(k * 2 + 1, chunk_size, -1)
        targets, feats = sample[k], np.concatenate(
            [sample[:k,:,:], sample[k+1:,:,:]], axis=0).reshape(2 * k * chunk_size, -1)
        feats, targets = torch.from_numpy(feats), torch.from_numpy(targets)
        
        return self.transform_fn(feats).to(torch.float), targets.to(torch.float)

    def __len__(self):
        return len(self.audio_id)


def create_dataloader(input, indices=None, is_random=True, scaler=None, transform_fn=lambda x: x, **kwargs):
    kwargs.setdefault('dataloader_args', 
        {'batch_size': 1024, 'num_workers': 8, 'shuffle': True})
    kwargs.setdefault('sample_args', {'k': 5, 'chunk_size': 96, 'alpha': 1.2})

    if is_random:
        _dataset = DEPADataset(
            input, indices=indices, scaler=scaler,\
            transform_fn=transform_fn, **kwargs['sample_args'])

    return DataLoader(_dataset, **kwargs['dataloader_args'])
