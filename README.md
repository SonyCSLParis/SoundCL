# Continual learning for Speech Command

This repository creates a basis upon wich one can try out continual learning strategies on the [Google Speech Command Dataset](http://arxiv.org/abs/1804.03209).

## The Dataset

The [Google Speech Command Dataset](http://arxiv.org/abs/1804.03209) is composed of 35 classes of 1 second sound utturences. This dataset regroups words usefull for robotics commands, and other words such as numbers and names.

## The framework

This implementation is based on [Avalanche](https://arxiv.org/abs/2104.00405) a python library based on pytorch adapted to continual learning. This enables us to try out different continual learning strategies and models easily.

More information on this library can be found in their [documentation](https://avalanche.continualai.org/) and [API](https://avalanche-api.continualai.org/en/v0.3.1/).

## Logging

Everything is logged with the python module sacred to a [MongoDB](https://www.mongodb.com/) database of your choosing.

To access this database and review/compare your experiments you can use a tool like [omniboard](https://github.com/vivekratnavel/omniboard).

## How to run

To run your experiment:

- Run the `setup.sh` script to create the necessary folders for the environment.
- Move into the code directory using `cd src`
- Set the name of your experience and your desired parameters in the `cfg()` function in `main.py`
- Run the experiment using `python3 main.py`

## Additionnal web tools

The `/web` directory contains two [Flask](https://flask.palletsprojects.com/en/2.3.x/) apps usefull for creating a dataset and testing models on a website. For more info please refer to the `/web/README.md` file.
