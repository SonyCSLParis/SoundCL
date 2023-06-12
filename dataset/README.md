# Datasets

This folder is meant to hold the datasets used for training.
At the moment two datasets are compatible with the repository:

- The [Speech Commands dataset](http://arxiv.org/abs/1804.03209)
- The [MLCommons dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/fe131d7f5a6b38b23cc967316c13dae2-Paper-round2.pdf)

## Download

### Speech Commands

The speech commands dataset can be automatically downloaded when running the code for the first time with the option `download=True`.

### ML Commons

To download the Ml commons dataset head to this [website](https://mlcommons.org/en/multilingual-spoken-words/). In this repo we've chosen to take a balanced subset of classes (in the english language) in order to have better training with our smaller models.

The scripts used for preprocessing and splitting the data can be found in the `./MLCommons` directory :

- `preprocess.ipynb` was used to chose a subset downsample and convert to .wav
- `augment.py` was used to add augmentations to random files in a balanced way.
- `split.py` was used to create the txt files for the test and validation split.
