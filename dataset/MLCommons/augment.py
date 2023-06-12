import os
import random
from tqdm import tqdm
from torch_audiomentations import Shift
from audio_augmentations import Noise,Gain
import torchaudio

data_dir = './subset2/'

classes = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

p=0.1

transforms=[Shift(min_shift=-0.2,max_shift=0.2,p=1),Noise(min_snr=0.1),Gain()]

for class_folder in classes:
    class_path= os.path.join(data_dir,class_folder)
    samples=os.listdir(class_path)
    nb_samples=int(len(samples)*p)
    random_samples= random.sample(samples,k=nb_samples)

    for file in tqdm(random_samples,desc=f"Augmenting file from {class_folder}"):
        file_path = os.path.join(class_path, file)

        waveform, sample_rate = torchaudio.load(file_path)

        waveform=waveform.unsqueeze(0)

        for id,transform in enumerate(transforms):
            augmented_waveform = transform(waveform)

            augmented_file_name = f"augmented_{id}_{file}"
            augmented_file_path = os.path.join(class_path, augmented_file_name)

            torchaudio.save(augmented_file_path, augmented_waveform.squeeze(0), sample_rate)