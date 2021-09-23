import yaml
import torch
import sys
from loguru import logger
import h5py
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import TimeMasking, FrequencyMasking
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pypeln.process as pr



def parse_config(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    return yaml_config

def genlogger(file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if file:
        logger.add(file, enqueue=True, format=log_format)
    return logger

def dataset_split(input, debug=False, random_state=0):
    with h5py.File(input, 'r') as input:
        indices = list(input.keys())
    if debug:
        indices = indices[: int(0.01 * len(indices))]
    train, test = train_test_split(indices,
        test_size=0.1, random_state=random_state)
    train, dev = train_test_split(train,
        test_size=0.1, random_state=random_state)
    
    return train, dev, test

# def normalization(input, indices, normalization=False, **kwargs):
#     kwargs.setdefault('with_mean', True)
#     kwargs.setdefault('with_std', True)

#     scaler = StandardScaler(**kwargs) if normalization else None
#     inputdim = 0
#     with h5py.File(input, 'r') as input:
#         for key in tqdm(indices, desc='Indices Traversal: '):
#             if scaler is not None:
#                 scaler.partial_fit(input[key][()])
#             inputdim = input[key][()].shape[-1] if not inputdim else inputdim
    
#     return scaler, inputdim

def get_transform(freq_mask_param=20, time_mask_param=10, p=0.8):
    time_mask_fn, freq_mask_fn = TimeMasking(time_mask_param),\
        FrequencyMasking(freq_mask_param)
    def transform_fn(spectrogram):
        if torch.rand(1).item() < p:
            return time_mask_fn(freq_mask_fn(spectrogram))
        return spectrogram
    return transform_fn



def normalization(input_file, indices, normalization=False, c=4, **kwargs):
    kwargs.setdefault('with_mean', True)
    kwargs.setdefault('with_std', True)

    scaler = None
    with h5py.File(input_file, 'r') as input:
        key = list(input.keys())[0]
        inputdim = input[key].shape[-1]
    if not normalization:
        return scaler, inputdim

    def calculate(indices):
        if not len(indices):
            return None
        scaler = StandardScaler(**kwargs)
        with h5py.File(input_file, 'r') as input:
            for key in indices:
                scaler.partial_fit(input[key][()])
        return scaler

    def merge_scaler(scaler_1, scaler_2, **kwargs):
        if scaler_1 is None or scaler_2 is None:
            return scaler_1 if scaler_1 else scaler_2

        scaler = StandardScaler(**kwargs)
        mean_1, mean_2 = scaler_1.mean_, scaler_2.mean_
        assert mean_1.shape[0] == mean_2.shape[0]
        scaler.n_features_in_ = mean_1.shape[0]

        var_1, var_2 = scaler_1.var_, scaler_2.var_
        N1, N2 = scaler_1.n_samples_seen_, scaler_2.n_samples_seen_
        scaler.n_samples_seen_ = N1 + N2

        mean = (N1 * mean_1 + N2 * mean_2) / (N1 + N2)
        scaler.mean_ = mean

        d_mean_1, d_mean_2 = mean_1 - mean, mean_2 - mean
        var = N1 * (var_1 + d_mean_1 ** 2) + N2 * (var_2 + d_mean_2 ** 2)
        var /= (N1 + N2)
        scaler.var_, scaler.scale_ = var, np.sqrt(var)

        return scaler

    chunk_size = len(indices) // c
    chunked_indices = [
        indices[i * chunk_size: (i + 1) * chunk_size] for i in range(9)]

    with tqdm(total=len(chunked_indices)) as pbar:
        for partial_scaler in pr.map(
                calculate, chunked_indices, workers=c, maxsize=c*2):
            scaler = merge_scaler(scaler, partial_scaler)\
                if scaler else partial_scaler
            pbar.update()
    
    return scaler, inputdim