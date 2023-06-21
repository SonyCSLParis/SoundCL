import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import math
import random
from kymatio.torch import Scattering1D

class AddWhiteNoise(nn.Module):
    """Transformation that adds white noise to the audio signal
    
    Example :

    >>> x = torch.zeros(16000)
    >>> transform = AddWhiteNoise()
    >>> x_with_noise = AddWhiteNoise(x)
    """
    def __init__(self):
        super().__init__()

    def add_white_noise(self,audio_tensor,min_snr_db=20,max_snr_db=90,STD_n=0.5):
        """Adds a random gaussian white noise to the audio_tensor input

        Args:
            audio_tensor (torch.tensor): 1 dimensional pytorch tensor
            min_snr_db (int, optional): minimum signal to noise ratio in dB. Defaults to 20.
            max_snr_db (int, optional): maximum signal to noise ratio in dB. Defaults to 90.
            STD_n (float, optional): Standard deviation of the gaussian distribution used to generate the noise. Defaults to 0.5.

        Returns:
            torch.tensor: tensor with noise
        """
        noise=np.random.normal(0, STD_n, audio_tensor.shape)
        noise_power = torch.from_numpy(noise).norm(p=2)
        audio_power = audio_tensor.norm(p=2)

        snr_db = random.randint(min_snr_db,max_snr_db)
        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / audio_power

        return (noise/scale+audio_tensor)/2
    
    def forward(self,x):
        return(self.add_white_noise(audio_tensor=x))
    
class MfccTransform(nn.Module):
    """Transformation that returns the Mel-frequency cepstral coefficients of an audio tensor
    
    Example :

    >>> x = torch.zeros(16000)
    >>> transform = MfccTransform()
    >>> specgram = MfccTransform(x)
    
    We can visualize the generated ceptrum with matplotlib using the following :

    >>> fig, axs = plt.subplots(1, 1)
    >>> axs.set_title(title or "Mel-frequency cepstrum")
    >>> axs.set_ylabel(ylabel)
    >>> axs.set_xlabel("frame")
    >>> im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    >>> fig.colorbar(im, ax=axs)
    >>> plt.show(block=False)
    """

    def __init__(self,sample_rate):
        super().__init__()
        self.sample_rate=sample_rate

    def mfcc_transform(self,audio_tensor,sample_rate,n_fft=512,n_mfcc=64,hop_length=10,mel_scale='htk'):
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
    
    def forward(self, x):
        if len(x.shape)>1:
            batch_size, num_samples = x.shape

            mfcc_features = []
            for i in range(batch_size):
                audio_tensor = x[i]  # Extract each audio tensor from the batch
                
                mfcc = self.mfcc_transform(audio_tensor=audio_tensor,sample_rate=self.sample_rate)
                mfcc_features.append(mfcc)

            mfcc_features = torch.stack(mfcc_features)        
            return mfcc_features.permute(0,2,1)
        else:
            return self.mfcc_transform(audio_tensor=x,sample_rate=self.sample_rate)

class SpecAugment(nn.Module):
    """Transformation that returns double time-masked and frequency-masked Mel-frequency cepstral coefficients of an audio tensor
    
    Example :

    >>> x = torch.zeros(16000)
    >>> transform = MfccTransform()
    >>> specgram = MfccTransform(x)

    We can visualize the modified ceptrum with matplotlib using the following :
 
    >>> fig, axs = plt.subplots(1, 1)
    >>> axs.set_title(title or "Mel-frequency cepstrum")
    >>> axs.set_ylabel(ylabel)
    >>> axs.set_xlabel("frame")
    >>> im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    >>> fig.colorbar(im, ax=axs)
    >>> plt.show(block=False)
    """
    def __init__(self):
        super().__init__()
    
    def spec_aug(self,tensor,time_mask=50,freq_mask=5,prob=0.8):
        time_masking = T.TimeMasking(time_mask_param=time_mask,p=prob)
        freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask)
        return freq_masking(freq_masking(time_masking(time_masking(tensor))))
    
    def forward(self,x):
        return self.spec_aug(tensor=x)

class Scattering(nn.Module):
    """Wrapper for kymatio's scattering transform. Returns the scattering coefficients of the input.
    
    For more information about the transform checkout : https://www.kymat.io/
    """
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
        x=self.scattering(x.squeeze(-1))
        #print(x.shape)
        x=torch.log(torch.abs(x)+self.log_eps)
        x=self.batch_norm(x.unsqueeze(1))
        return x.squeeze(1).permute(0,2,1)
