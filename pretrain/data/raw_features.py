import h5py
import librosa
from tqdm import tqdm
import pypeln.process as pr
import argparse
import numpy as np
from glob import glob
import os
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', type=str, help="path of raw wav files")
parser.add_argument('-t', '--type', type=str, help="type of input features")
parser.add_argument('--config', type=str, help='configuration of audio features')
parser.add_argument('-c', type=int, default=20, help='number of calculating cores')
parser.add_argument('--sample_rate', type=int, default=22050, help='sample rate')
parser.add_argument('-o', '--output', type=str, default='stft.hdf5', help='output feature file')

args = parser.parse_args()

def parse_config(config_file):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    return yaml_config

audio_config = parse_config(args.config)[args.type]
files = glob(os.path.join(args.path, '*.wav'))


def get_transform_func(type='stft'):
    if type == 'stft':
        def stft_transform(raw_wav, **kwargs):
            kwargs['win_length'] = (kwargs.get('win_length', 20) * sr) // 1000
            kwargs['hop_length'] = (kwargs.get('hop_length', 5) * sr) // 1000
            return np.log(np.abs(librosa.stft(raw_wav, **kwargs)).T + 1e-12)
        return stft_transform
    elif type == 'lms':
        def lms_transform(raw_wav, **kwargs):
            kwargs['hop_length'] = kwargs.get('hop_length', 5) * sr // 1000
            return np.log(librosa.feature.melspectrogram(raw_wav, **kwargs).T + 1e-12)
        return lms_transform
    else:
        def mfcc_transform(raw_wav, **kwargs):
            return librosa.feature.mfcc(raw_wav, **kwargs)
        return mfcc_transform

transform_fn = get_transform_func(args.type)
sr = args.sample_rate
if args.type != 'stft':
    audio_config['sr'] = sr

def read_wav(file):
    audio, _ = librosa.load(file, sr=sr, duration=7200)
    index = os.path.split(file)[-1].split('.')[0]
    features = transform_fn(audio, **audio_config)
    return index, features



with h5py.File(args.output, 'w') as output, tqdm(total=len(files)) as pbar:
    for index, features in pr.map(read_wav, files, args.c, args.c*2):
        output[str(index)] = features
        pbar.update()

        
