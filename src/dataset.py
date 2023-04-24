from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.datasets import default_dataset_location
import torch
import torch.nn as nn
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio.transforms as T
import numpy as np
import math
import random


def speech_commands_collate(batch):
    """Collate function for setting up the dataloader

    Args:
        batch (int): batch size

    Returns:
        batch: return batched data in the form ; audio_tensor,target,task_label
    """
    tensors, targets, t_labels = [], [], []
    for waveform, label, rate, sid, uid, t_label in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]
        t_labels += [torch.tensor(t_label)]
    tensors = [item.t() for item in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.0
    )
    if len(tensors.size()) == 2:  # add feature dimension
        tensors = tensors.unsqueeze(-1)
    targets = torch.stack(targets)
    t_labels = torch.stack(t_labels)
    return tensors, targets, t_labels# Fix for convolution.permute(0,2,1)

def spec_aug(tensor,time_mask=50,freq_mask=5,prob=0.8):
    time_masking = T.TimeMasking(time_mask_param=time_mask,p=prob)
    freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask)
    return freq_masking(freq_masking(time_masking(time_masking(tensor))))
    
def mfcc_transform(audio_tensor,sample_rate,n_fft=512,n_mfcc=64,hop_length=10,mel_scale='htk'):
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "hop_length": hop_length,
            "mel_scale": mel_scale,
            "n_fft": n_fft,
        },
    )
    return transform(audio_tensor.to(dtype=torch.float32))
    
def add_white_noise(audio_tensor,min_snr_db=20,max_snr_db=90,STD_n=0.5,norm=2):
    noise=np.random.normal(0, STD_n, audio_tensor.shape)
    noise_power = torch.from_numpy(noise).norm(p=norm)
    audio_power = audio_tensor.norm(p=norm)

    snr_db = random.randint(min_snr_db,max_snr_db)
    snr = math.exp(snr_db / 10)
    scale = snr * noise_power / audio_power

    return (noise/scale+audio_tensor)/2

class AddWhiteNoise(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        #print("ONE",x.shape)
        return(add_white_noise(audio_tensor=x))
    
class MfccTransform(nn.Module):
    def __init__(self,sample_rate):
        super().__init__()
        self.sample_rate=sample_rate
    def forward(self,x):
        #print("TWO",x.shape)
        return mfcc_transform(audio_tensor=x,sample_rate=self.sample_rate)

class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        #print(x.shape)
        return spec_aug(tensor=x)


class SpeechCommandsData(SPEECHCOMMANDS):
    def __init__(self, root, url, download, subset):
        super().__init__(root=root, download=download, subset=subset, url=url)
        self.labels_names = [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]

    def __getitem__(self, item):
        wave, rate, label, speaker_id, ut_number = super().__getitem__(item)
        label = self.labels_names.index(label)
        wave = wave.squeeze(0)  # (T,)
        return wave, label, rate, speaker_id, ut_number
    
class Audio_Dataset():
    def __init__(self):
        
        self.train_transformation = nn.Sequential(AddWhiteNoise(),MfccTransform(sample_rate=16000))#,SpecAugment())
        self.train_target_transformation = None
        
        self.test_transformation = nn.Sequential(MfccTransform(sample_rate=16000))
        self.test_target_transformation = None
        self.labels_names = [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]
        self.tranform_groups = {
            'train':(self.train_transformation,self.train_target_transformation),
            'eval':(self.test_transformation,self.test_target_transformation)
        }

    def SpeechCommands(
        root=default_dataset_location("speechcommands"),
        url="speech_commands_v0.02",
        download=True,
        subset=None,
        transforms=None
    ):
        """
        root: dataset root location
        url: version name of the dataset
        download: automatically download the dataset, if not present
        subset: one of 'training', 'validation', 'testing'
        """
        
        dataset = SpeechCommandsData(
            root='../dataset/',
            download=download,
            subset=subset,
            url=url,
        )
        labels = [datapoint[1] for datapoint in dataset]

        return make_classification_dataset(
            dataset, collate_fn=speech_commands_collate, targets=labels,
            transform_groups=transforms
        )

    def __call__(self,train):
        """Function call to AudioDataset

        Args:
            train (bool): True for training subset and False for testing

        Returns:
            ClassificationDataset: avalanche comatible speech command dataset
        """
        if train:
            return self.SpeechCommands(subset='training',transforms=self.tranform_groups)
        else:
            return self.SpeechCommands(subset='testing',transforms=self.tranform_groups)