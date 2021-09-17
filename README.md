# DEPA: Self-Supervised Audio Embedding for Depression Detection
This is the source code for DEPA.


The code is centered around the config files placed at `pretrain/config/config_*.yaml`.

# Extracting pretrain features
Use `pretrain/data/raw_features.py` scripts to extract LMS, STFT, or MFCC features for pretraining.
You can modify `pretrain/config/config_features.yaml` to modify the configuration of features

# Running the code

The main script of this repo is `pretrain/pretrain.py`.

`pretrain.py` the following options ( ran as `python pretrain.py OPTION`):
* `train`: Trains a model given a config file (default is `pretrain/config/config_pretrain.yaml`).
* `encoding`: Extracting DEPA given model path and input file and output to the given output file.
* `Pipeline`: Conduct the above two steps

Example: 
```python
python pretrain.py train pretrain/config/config_pretrain.yaml False # mean not debug
```