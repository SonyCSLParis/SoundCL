# Continual learning for Speech Command

This repository creates a basis upon wich one can try out continual learning strategies on the google Speech Command Dataset.

## Logging
Everything is logged with the python module sacred to a [MongoDB](https://www.mongodb.com/) database of your choosing.

To access this database and review/compare your experiments you can use a tool like [omniboard](https://github.com/vivekratnavel/omniboard).
## How to run
To run your experiment:
- Move into the code directory using `cd src`
- Set the name of your experience and your desired parameters in the `cfg()` function in `main.py`
- Run the experiment using `python3 main.py`
