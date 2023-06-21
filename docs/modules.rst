src
===

The **src** folder contains all the scripts and modules used to train continual learning models using the Avalanche library.

This folder is organized into the following distinct components:

- **main :** The main script that runs the show, utilizing Avalanche and Sacred for logging.
- **dataset :** Wrappers for datasets.
- **infer :** A basic inference script for live testing the models.
- **transforms :** Wrappers and PyTorch implementations of audio transformations.
- **models :** A collection of different PyTorch models used for testing.
- **matchbox :** An implementation of MatchboxNet in PyTorch.
- **plugins :** Personalized Avalanche plugins used for training.
- **templates :** Personalized Avalanche templates used for training.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


   main
   dataset
   infer
   transforms
   models
   matchbox
   plugins
   templates
