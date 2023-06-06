from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.datasets import default_dataset_location
import torch
import torch.nn as nn
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
import torchaudio.transforms as T
import numpy as np
import math
import random
import h5py
import os
import typing
from pathlib import Path
from tqdm import tqdm
import logging
from kymatio.torch import Scattering1D

def speech_commands_collate(batch):
    """Collate function for setting up the dataloader

    Args:
        batch (int): batch size

    Returns:
        batch: return batched data in the form ; audio_tensor,target,task_label
    """
    tensors, targets, t_labels = [], [], []
    for waveform, label, rate, sid, uid, t_label in batch:
        if isinstance(waveform,np.ndarray):
            tensors += [torch.from_numpy(waveform)]
        elif isinstance(waveform, torch.Tensor):
            tensors += [waveform]
        else:
            raise ValueError("Waveform must be saved as torch.tensor or np.array")
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

def preprocess_and_save_dataset(dataset, save_path,transformation):
    if os.path.isfile(save_path):
        # The preprocessed dataset already exists, no need to preprocess again
        logging.info("The preprocessed dataset already exists, using cache")
        return
    
    # Create a new HDF5 file to store the preprocessed data
    with h5py.File(save_path, 'w') as f:
        # Create HDF5 datasets for the waveform, rate, label, speaker_id, and ut_number
        waveform_dset = f.create_dataset('waveform', shape=(len(dataset),64,1601), dtype=np.dtype('float32'))
        rate_dset = f.create_dataset('rate', shape=(len(dataset),), dtype='int32')
        label_dset = f.create_dataset('label', shape=(len(dataset),), dtype='int32')
        speaker_id_dset = f.create_dataset('speaker_id', shape=(len(dataset),), dtype=h5py.special_dtype(vlen=str))
        ut_number_dset = f.create_dataset('ut_number', shape=(len(dataset),), dtype='int32')

        # Preprocess and save each item in the dataset
        with tqdm(total=len(dataset),desc="Preprocessing dataset") as pbar:
            for i, item in enumerate(dataset):
                wave, label, rate, speaker_id, ut_number = item
               
                # Apply preprocessing to the waveform
                wave = torch.nn.functional.pad(input=wave,pad=[0,16000-wave.shape[0]],mode='constant', value=0)
                wave = transformation(wave)

                # Add the preprocessed data to the HDF5 datasets
                waveform_dset[i] = wave.numpy()
                rate_dset[i] = rate
                label_dset[i] = label
                speaker_id_dset[i] = speaker_id
                ut_number_dset[i] = ut_number

                # Update the progress bar
                pbar.update(1)

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
            "n_mels":64,
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
        return(add_white_noise(audio_tensor=x))
    
class MfccTransform(nn.Module):
    def __init__(self,sample_rate):
        super().__init__()
        self.sample_rate=sample_rate
    def forward(self, x):
        if len(x.shape)>1:
            batch_size, num_samples = x.shape

            mfcc_features = []
            for i in range(batch_size):
                audio_tensor = x[i]  # Extract each audio tensor from the batch
                
                mfcc = mfcc_transform(audio_tensor=audio_tensor,sample_rate=self.sample_rate)
                mfcc_features.append(mfcc)

            mfcc_features = torch.stack(mfcc_features)        
            return mfcc_features.permute(0,2,1)
        else:
            return mfcc_transform(audio_tensor=x,sample_rate=self.sample_rate)

class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return spec_aug(tensor=x)

class Scattering(nn.Module):
    def __init__(self):
        super().__init__()
        #Scattering hyperparameters
        T=16000
        J=4
        Q=8
        self.log_eps=1e-6
        #Layers
        self.scattering= Scattering1D(J=J,shape=T,Q=Q)
        self.batch_norm= nn.BatchNorm2d(1)
    def forward(self,x):
        #print(x.shape)
        x=self.scattering(x.squeeze())
        #print(x.shape)
        x=torch.log(torch.abs(x)+self.log_eps)
        x=self.batch_norm(x.unsqueeze(1))
        #print(x.shape)
        return x.squeeze(1).permute(0,2,1)

class SpeechCommandsData(SPEECHCOMMANDS):
    def __init__(self, root, url, download, subset):
        super().__init__(root=root, download=download, subset=subset, url=url)
        self.labels_names = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]


    def __getitem__(self, item):
        wave, rate, label, speaker_id, ut_number = super().__getitem__(item)

        label = self.labels_names.index(label)
        wave = wave.squeeze(0)  # (T,)

        return wave, label, rate, speaker_id, ut_number

class MLcommonsData():
    def __init__(self, root,sub_folder,subset,folder_in_archive = "MLCommons"):
        if sub_folder=='subset1':
            self.labels_names = ['about', 'books', 'car', 'county', 'different', 'door', 'felt', 'game', 'has', 'live', 'man', 'north', 'open', 'party', 'put', 'run', 'side', 'sun', 'thing', 'trying', 'who', 'words', 'back', 'boy', 'church', 'day', 'does', 'end', 'friend', 'general', 'here', 'love', 'need', 'now', 'our', 'people', 'river', 'service', 'song', 'sure', 'then', 'treasure', 'why', 'you']
        elif sub_folder=='subset2':
            self.labels_names= ["tried","hey","career","south","please","working","building","old","around","company","himself","language","album","family","young","returned","important","throughout","understand","include","business","daughter","everything","englishman","between","outside",'about', 'books', 'car', 'county', 'different', 'door', 'felt', 'game', 'has', 'live', 'man', 'north', 'open', 'party', 'put', 'run', 'side', 'sun', 'thing', 'trying', 'who', 'words', 'back', 'boy', 'church', 'day', 'does', 'end', 'friend', 'general', 'here', 'love', 'need', 'now', 'our', 'people', 'river', 'service', 'song', 'sure', 'then', 'treasure', 'why', 'you']
        else:
            raise NotImplemented()

        if subset is not None and subset not in ["training", "validation", "testing"]:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)

        basename = os.path.basename(sub_folder)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path or set `download=True` to download it"
            )

        if subset == "validation":
            self._walker = self._load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = self._load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(self._load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [
                w
                for w in walker
                if os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker]

    def _load_list(self,root, *filenames):
        output = []
        for filename in filenames:
            filepath = os.path.join(root, filename)
            with open(filepath) as fileobj:
                output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
        return output
    
    def get_metadata(self, n: int) -> typing.Tuple[str, int, str, str, int]:
        relpath = os.path.relpath(self._walker[n], self._archive)
        reldir, filename = os.path.split(relpath)
        _, label = os.path.split(reldir)
        return relpath, 16000, label
    
    def __getitem__(self, n: int) -> typing.Tuple[torch.Tensor, int, str, str, int]:
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        return waveform.squeeze(0),self.labels_names.index(metadata[2]),metadata[1],0,0
    
    def __len__(self) -> int:
        return len(self._walker)


class CachedAudio(Dataset):
    def __init__(self,subset,train_cache_path='../dataset_cache/',test_cache_path='../dataset_cache/'):
        self.labels_names = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]
        self.subset=subset
        self.preprocessed_train_path = os.path.join(train_cache_path, "preprocessed_train.h5")
        self.preprocessed_test_path = os.path.join(test_cache_path, "preprocessed_test.h5")

    def __len__(self):
        if self.subset=='training':
            return h5py.File(self.preprocessed_train_path, 'r')['waveform'].len()
        elif self.subset=='testing':
            return h5py.File(self.preprocessed_test_path, 'r')['waveform'].len()
        else:
            raise ValueError("Unknown data subset. Choose from : training or testing.")


    def __getitem__(self,item):
        if self.subset=='training':
            # Load the preprocessed dataset
            with h5py.File(self.preprocessed_train_path, 'r') as f:
                wave = f['waveform'][item]
                rate = f['rate'][item]
                label = f['label'][item]
                speaker_id = f['speaker_id'][item]
                ut_number = f['ut_number'][item]
        elif self.subset=='testing':
            # Load the preprocessed dataset
            with h5py.File(self.preprocessed_test_path, 'r') as f:
                wave = f['waveform'][item]
                rate = f['rate'][item]
                label = f['label'][item]
                speaker_id = f['speaker_id'][item]
                ut_number = f['ut_number'][item]
        else:
            raise ValueError("Unknown data subset. Choose from : training or testing.")
        return wave, label, rate, speaker_id, ut_number
    
class Audio_Dataset():
    def __init__(self):
        
        self.labels_names = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]
        
        self.commons_names = ['about', 'books', 'car', 'county', 'different', 'door', 'felt', 'game', 'has', 'live', 'man', 'north', 'open', 'party', 'put', 'run', 'side', 'sun', 'thing', 'trying', 'who', 'words', 'back', 'boy', 'church', 'day', 'does', 'end', 'friend', 'general', 'here', 'love', 'need', 'now', 'our', 'people', 'river', 'service', 'song', 'sure', 'then', 'treasure', 'why', 'you']
        self.commons_names2= ["tried","hey","career","south","please","working","building","old","around","company","himself","language","album","family","young","returned","important","throughout","understand","include","business","daughter","everything","englishman","between","outside",'about', 'books', 'car', 'county', 'different', 'door', 'felt', 'game', 'has', 'live', 'man', 'north', 'open', 'party', 'put', 'run', 'side', 'sun', 'thing', 'trying', 'who', 'words', 'back', 'boy', 'church', 'day', 'does', 'end', 'friend', 'general', 'here', 'love', 'need', 'now', 'our', 'people', 'river', 'service', 'song', 'sure', 'then', 'treasure', 'why', 'you']

        self.train_transformation = MfccTransform(sample_rate=16000)#nn.Sequential(AddWhiteNoise(),MfccTransform(sample_rate=16000))#,SpecAugment())
        self.test_transformation = MfccTransform(sample_rate=16000)#nn.Sequential(MfccTransform(sample_rate=16000))
        self.transform_groups={
            'train':(self.train_transformation,None),
            'test':(self.test_transformation,None)
        }

    def SpeechCommands(self,
        root=default_dataset_location("speechcommands"),
        url="speech_commands_v0.02",
        download=True,
        subset=None,
        transforms=None,
        pre_process=True
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

        if pre_process:

            if subset=='training':
                self.preprocessed_train_path = os.path.join('../dataset_cache/', "preprocessed_train.h5")
                preprocess_and_save_dataset(dataset=dataset, save_path=self.preprocessed_train_path,transformation=self.train_transformation)
                cached_dataset=CachedAudio(subset=subset)
                labels = [datapoint[1] for datapoint in cached_dataset]
                
            elif subset=='testing':
                
                self.preprocessed_test_path = os.path.join('../dataset_cache/', "preprocessed_test.h5")
                preprocess_and_save_dataset(dataset=dataset, save_path=self.preprocessed_test_path,transformation=self.test_transformation)
                cached_dataset=CachedAudio(subset=subset)
                labels = [datapoint[1] for datapoint in cached_dataset]

            else:
                raise ValueError("Unknown data subset. Choose from : training or testing.")

            return make_classification_dataset(cached_dataset, collate_fn=speech_commands_collate, targets=labels
                                               )
        else:
            labels = [datapoint[1] for datapoint in dataset]

            return make_classification_dataset(dataset, collate_fn=speech_commands_collate, targets=labels,transform_groups=None)

    def MLCommons(self,
        root='../dataset/',
        sub_folder="subset1",
        subset="training",
        transforms=None,):

        #Because the Ml commons dataset has the same structure as the speech commands dataset we can use the same wrapper
        # we create empty validation and testing list because this is only used for pretraining
        dataset = MLcommonsData(
            root='../dataset/',
            sub_folder=sub_folder,
            subset=subset,
        )

        labels = [datapoint[1] for datapoint in dataset]

        return make_classification_dataset(dataset, collate_fn=speech_commands_collate, targets=labels,transform_groups=None)


    def __call__(self,train,pre_process):
        """Function call to AudioDataset

        Args:
            train (bool): True for training subset and False for testing
            pre_process (bool): Preprocess all the dataset before or do the preprocessing on the fly

        Returns:
            ClassificationDataset: avalanche comatible speech command dataset
        """
        if train:
            return self.SpeechCommands(subset='training',pre_process=pre_process,transforms=self.transform_groups)
        else:
            return self.SpeechCommands(subset='testing',pre_process=pre_process,transforms=self.transform_groups)
    