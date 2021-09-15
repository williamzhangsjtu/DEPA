import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import h5py



def parse_config(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    return yaml_config

def genlogger(outputfile):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(logging.INFO)
    stdlog = logging.StreamHandler(sys.stdout)
    stdlog.setFormatter(formatter)
    file_handler = logging.FileHandler(outputfile)
    file_handler.setFormatter(formatter)
    # Log to stdout
    logger.addHandler(file_handler)
    logger.addHandler(stdlog)
    return logger

def dataset_split(input, random_state=0):
    with h5py.File(input, 'r') as input:
        indices = input.keys()
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
        for key in input.keys():
            if scaler is not None:
                scaler.partial_fit(input[key][()])
            inputdim = input[key][()].shape[-1] if not inputdim else inputdim
    
    return scaler, inputdim