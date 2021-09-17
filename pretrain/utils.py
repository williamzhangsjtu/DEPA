import yaml
import torch
import sys
from loguru import logger
import h5py
import numpy as np
from torchaudio.transforms import TimeMasking, FrequencyMasking
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



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
        indices = indices[: int(0.02 * len(indices))]
    train, test = train_test_split(indices,
        test_size=0.1, random_state=random_state)
    train, dev = train_test_split(train,
        test_size=0.1, random_state=random_state)
    
    return train, dev, test

def normalization(input, indices, normalization=True, **kwargs):
    kwargs.setdefault('with_mean', True)
    kwargs.setdefault('with_std', True)

    scaler = StandardScaler(**kwargs) if normalization else None
    inputdim = 0
    with h5py.File(input, 'r') as input:
        for key in indices:
            if scaler is not None:
                scaler.partial_fit(input[key][()])
            inputdim = input[key][()].shape[-1] if not inputdim else inputdim
    
    return scaler, inputdim

def get_transform(freq_mask_param=20, time_mask_param=10, p=0.8):
    time_mask_fn, freq_mask_fn = TimeMasking(time_mask_param),\
        FrequencyMasking(freq_mask_param)
    def transform_fn(spectrogram):
        if torch.rand(1).item() < p:
            return time_mask_fn(freq_mask_fn(spectrogram))
        return spectrogram
    return transform_fn