import torch
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.utils import _load_waveform

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.datasets import default_dataset_location

import numpy as np

import os
import h5py
import typing
import logging
from tqdm import tqdm
from pathlib import Path


def speech_commands_collate(batch):
    """Collate function for setting up the dataloader

    Args:
        batch (int): batch size

    Returns:
        batch: return batched data in the form ; audio_tensor,target,task_label
    """
    #FIXME with precached data we seem to always take this first loop
    if len(batch)==1:
        waveform, label, rate, sid, uid, t_label = batch[0]
        waveform=waveform.squeeze()
        tensor_size=waveform.size(0)
        size=128
        if tensor_size < size:
            padding_size = size - tensor_size
            padded_tensor = torch.cat((waveform, torch.zeros(padding_size)), dim=0)
            return padded_tensor.unsqueeze(0),torch.tensor(label).unsqueeze(0),torch.tensor(t_label).unsqueeze(0)
        elif tensor_size > size:
            cut_tensor = waveform[:size]
            return cut_tensor.unsqueeze(0),torch.tensor(label).unsqueeze(0),torch.tensor(t_label).unsqueeze(0)
        else:
            return waveform.unsqueeze(0),torch.tensor(label).unsqueeze(0),torch.tensor(t_label).unsqueeze(0)

    else:
        
        tensors, targets, t_labels = [], [], []

        for waveform, label, rate, sid, uid, t_label in batch:#FIXME this is only a temporary solution for icarl 

            if isinstance(waveform,np.ndarray):
                waveform = torch.from_numpy(waveform)
            elif isinstance(waveform, torch.Tensor):
                pass
            else:
                raise ValueError("Waveform must be saved as torch.tensor or np.array")
            
            tensor_size=waveform.size(0)
            size=16000
            
            if tensor_size < size:
                padding_size = size - tensor_size
                waveform = torch.cat((waveform, torch.zeros(padding_size)), dim=0)
            elif tensor_size > size:
                waveform = waveform[:size]
        
            tensors += [waveform]

            targets += [torch.tensor(label)]
            t_labels += [torch.tensor(t_label)]

        tensors = [item.t() for item in tensors]
        tensors = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0.0
        )
        #if len(tensors.size()) == 2:  # add feature dimension
        #    tensors = tensors.unsqueeze(-1)
        targets = torch.stack(targets)
        t_labels = torch.stack(t_labels)
        return tensors, targets, t_labels# Fix for convolution.permute(0,2,1)

@torch.no_grad()
def preprocess_and_save_dataset(dataset, save_path : str,transformation,output_shape=[]):
    """Function for preprocessing and saving datasets.

    .. important::
        This function only works for the SpeechCommands dataset, or for dataset that have those specific entries :  wave, label, rate, speaker_id, utterance_number

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be processed
        save_path (str): Save path for the preprocessed dataset
        transformation (torch.nn.Module): The transformations that will be applied to the data
        output_shape (list): Output shape of an element of the transformation. Defaults to [].

    Raises:
        AttributeError: If given output shape is not a list or is an empty list
    """
    
    if os.path.isfile(save_path):
        # The preprocessed dataset already exists, no need to preprocess again
        logging.info("The preprocessed dataset already exists, using cache")
        return
    
    if output_shape==[] or not isinstance(output_shape,list):
        raise AttributeError("Specify the shape of the output of the transform using a list")

    # Create a new HDF5 file to store the preprocessed data
    with h5py.File(save_path, 'w') as f:

        # Create HDF5 datasets for the waveform, rate, label, speaker_id, and ut_number
        waveform_dset = f.create_dataset('waveform', shape=(len(dataset),*output_shape), dtype=np.dtype('float32'))
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
                wave = transformation(wave.unsqueeze(0)).squeeze(0)

                # Add the preprocessed data to the HDF5 datasets
                waveform_dset[i] = wave.numpy()
                rate_dset[i] = rate
                label_dset[i] = label
                speaker_id_dset[i] = speaker_id
                ut_number_dset[i] = ut_number

                # Update the progress bar
                pbar.update(1)

    

class SpeechCommandsData(SPEECHCOMMANDS):
    """Wrapper for torchaudio's speechcommand dataset.
    """
    def __init__(self, root, url, download, subset):
        super().__init__(root=root, download=download, subset=subset, url=url)
        self.labels_names = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]


    def __getitem__(self, item):
        wave, rate, label, speaker_id, ut_number = super().__getitem__(item)

        label = self.labels_names.index(label)
        wave = wave.squeeze(0)  # (T,)

        return wave, label, rate, speaker_id, ut_number

class MLcommonsData():
    """Wrapper for a subset of the MlCommons `Multilingual Spoken Words dataset <https://mlcommons.org/en/multilingual-spoken-words/>`_
    """
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
    """Wrapper for cached `hdf5 <https://www.h5py.org/>`_ audio datasets.
    """
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
        
        return torch.from_numpy(wave), label, rate, speaker_id, ut_number
    
class Audio_Dataset():
    """Avalanche audio datasets wrapper.
    """
    def __init__(self,train_transformation=None,test_transformation=None):
        
        self.labels_names  = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]
        self.commons_names = ['about', 'books', 'car', 'county', 'different', 'door', 'felt', 'game', 'has', 'live', 'man', 'north', 'open', 'party', 'put', 'run', 'side', 'sun', 'thing', 'trying', 'who', 'words', 'back', 'boy', 'church', 'day', 'does', 'end', 'friend', 'general', 'here', 'love', 'need', 'now', 'our', 'people', 'river', 'service', 'song', 'sure', 'then', 'treasure', 'why', 'you']
        self.commons_names2= ["tried","hey","career","south","please","working","building","old","around","company","himself","language","album","family","young","returned","important","throughout","understand","include","business","daughter","everything","englishman","between","outside",'about', 'books', 'car', 'county', 'different', 'door', 'felt', 'game', 'has', 'live', 'man', 'north', 'open', 'party', 'put', 'run', 'side', 'sun', 'thing', 'trying', 'who', 'words', 'back', 'boy', 'church', 'day', 'does', 'end', 'friend', 'general', 'here', 'love', 'need', 'now', 'our', 'people', 'river', 'service', 'song', 'sure', 'then', 'treasure', 'why', 'you']

        self.train_transformation = train_transformation
        self.test_transformation = test_transformation
        self.transform_groups={
            'train':(self.train_transformation,None),
            'eval':(self.test_transformation,None)
        }

    def SpeechCommands(self,
        root=default_dataset_location("speechcommands"),
        url="speech_commands_v0.02",
        download=True,
        subset=None,
        transforms=None,
        pre_process=True,
        output_shape=[],
    ):
        """SpeechCommands dataset wrapper function for avalanche lib.

        Args:
            root (str, optional): dataset root location. Defaults to default_dataset_location("speechcommands").
            url (str, optional): version name of the dataset. Defaults to "speech_commands_v0.02".
            download (bool, optional): automatically download the dataset, if not present. Defaults to True.
            subset (str, optional): one of 'training', 'validation', 'testing'. Defaults to None.
            transforms (torch.nn.Module, optional): transformations applied to the data. Defaults to None.
            pre_process (bool, optional): Enable prior preprocessing and saving of the dataset. Defaults to True.
            output_shape (list) : Output shape of a transformed element.

        Raises:
            ValueError: If an unkown subset is chosen

        Returns:
            ClassificationDataset: Avalanche's classification dataset
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
                preprocess_and_save_dataset(dataset=dataset, save_path=self.preprocessed_train_path,transformation=self.train_transformation,output_shape=output_shape)
                cached_dataset=CachedAudio(subset=subset)
                labels = [datapoint[1] for datapoint in cached_dataset]
                
            elif subset=='testing':
                
                self.preprocessed_test_path = os.path.join('../dataset_cache/', "preprocessed_test.h5")
                preprocess_and_save_dataset(dataset=dataset, save_path=self.preprocessed_test_path,transformation=self.test_transformation,output_shape=output_shape)
                cached_dataset=CachedAudio(subset=subset)
                labels = [datapoint[1] for datapoint in cached_dataset]

            else:
                raise ValueError("Unknown data subset. Choose from : training or testing.")

            return make_classification_dataset(cached_dataset, collate_fn=speech_commands_collate, targets=labels)
        
        else:
            labels = [datapoint[1] for datapoint in dataset]

            return make_classification_dataset(dataset, collate_fn=speech_commands_collate, targets=labels,transform_groups=self.transform_groups)

    def MLCommons(self,
        root='../dataset/',
        sub_folder="subset2",
        subset="training",
        transforms=None,):
        """MLCommons dataset wrapper function for avalanche lib.

        Args:
            root (str, optional):  dataset root location. Defaults to '../dataset/'.
            sub_folder (str, optional): dataset subset. Defaults to "subset2".
            subset (str, optional): one of 'training', 'validation', 'testing'. Defaults to "training".
            transforms (_type_, optional): transformations applied to the data. Defaults to None.

        Returns:
            ClassificationDataset: Avalanche's classification dataset
        """

        #Because the Ml commons dataset has the same structure as the speech commands dataset we can use the same wrapper
        # we create empty validation and testing list because this is only used for pretraining
        dataset = MLcommonsData(
            root='../dataset/',
            sub_folder=sub_folder,
            subset=subset,
        )

        labels = [datapoint[1] for datapoint in dataset]

        return make_classification_dataset(dataset, collate_fn=speech_commands_collate, targets=labels,transform_groups=self.transform_groups)


    def __call__(self,train,pre_process,output_shape=[]):
        """Function call to AudioDataset

        Args:
            train (bool): True for training subset and False for testing
            pre_process (bool): Preprocess all the dataset before or do the preprocessing on the fly
            output_shape (list) : Output shape of a transformed element.

        Returns:
            ClassificationDataset: avalanche comatible speech command dataset
        """
        if train:
            return self.SpeechCommands(subset='training',pre_process=pre_process,transforms=self.transform_groups,output_shape=output_shape)
        else:
            return self.SpeechCommands(subset='testing',pre_process=pre_process,transforms=self.transform_groups,output_shape=output_shape)
    