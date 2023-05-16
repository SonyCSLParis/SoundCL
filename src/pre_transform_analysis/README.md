# Transform analysis

The goal of this subfolder is to test the difference between the MFCC transform, and the Scattering transform for the preprocessing.

In the `Comparaison.ipynb` folder you will find two tests:

- The first was to map the direct output of the transform onto a lower manifold using UMAP and compare them
- The second was to check for robustness to variations

## Robustness

All of the scripts present in this folder are used for the robustness test:

- We split our main dataset into a male and female dataset using a fine-tuned wav2vec model in `gender_separation.py`
- We fixed the `txt` files containing the audio paths in each respective dataset in `subset_separation.py`
- We finally used trained models on those seperate dataset using the two different transforms and tested them on the opposite dataset in `Comparaison.ipynb`
