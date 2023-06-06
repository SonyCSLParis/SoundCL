import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models.generator import Generator

from matchbox.ConvASRDecoder import ConvASRDecoderClassification
from matchbox.ConvASREncoder import ConvASREncoder

from torchinfo import summary

from kymatio.torch import Scattering1D

class EncDecBaseModel(nn.Module):

    def __init__(self, num_mels, 
                 final_filter,
                 num_classes,
                 input_length):
        """Encoder decoder model for MatchboxNet from the paper : http://arxiv.org/abs/2004.08531

        Args:
            num_mels (int): number of mel features in the mfcc transform preprocessing
            final_filter (int): size of final conv filter in the encoder
            num_classes (int): number of output classes for classification
            input_length (int): input time dimension length
        Summary:
            ===========================================================================
            Layer (type:depth-idx)                             Param #
            ===========================================================================
            EncDecBaseModel                                    --
            ├─ConvASREncoder: 1-1                              --
            │    └─ConvBlock: 2-1                              --
            │    │    └─ModuleList: 3-1                        9,152
            │    │    └─Sequential: 3-2                        --
            │    └─ConvBlock: 2-2                              --
            │    │    └─ModuleList: 3-3                        9,984
            │    │    └─ModuleList: 3-4                        8,320
            │    │    └─Sequential: 3-5                        --
            │    └─ConvBlock: 2-3                              --
            │    │    └─ModuleList: 3-6                        5,184
            │    │    └─ModuleList: 3-7                        4,224
            │    │    └─Sequential: 3-8                        --
            │    └─ConvBlock: 2-4                              --
            │    │    └─ModuleList: 3-9                        5,312
            │    │    └─ModuleList: 3-10                       4,224
            │    │    └─Sequential: 3-11                       --
            │    └─ConvBlock: 2-5                              --
            │    │    └─ModuleList: 3-12                       10,304
            │    │    └─Sequential: 3-13                       --
            │    └─ConvBlock: 2-6                              --
            │    │    └─ModuleList: 3-14                       16,768
            │    │    └─Sequential: 3-15                       --
            ├─ConvASRDecoderClassification: 1-2                --
            │    └─AdaptiveAvgPool1d: 2-8                      --
            │    └─Sequential: 2-9                             --
            │    │    └─Linear: 3-22                           4,515
            ===========================================================================
            Total params: 77,987
            Trainable params: 77,987
            Non-trainable params: 0
            ===========================================================================
        """
        super(EncDecBaseModel, self).__init__()

        self.input_length = torch.tensor(input_length)

        self.encoder = ConvASREncoder(feat_in = num_mels)
        self.decoder = ConvASRDecoderClassification(feat_in = final_filter, num_classes= num_classes,return_logits=True)

    def forward(self, input_signal):
        encoded, encoded_len = self.encoder(audio_signal=input_signal, length=self.input_length)
        logits = self.decoder(encoder_output=encoded)
        return logits

#TODO investigate parameter number difference

class AudioVAE(nn.Module,Generator):
    def __init__(self, imgChannels=1, featureDim=15656, zDim=256):
        super(AudioVAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(in_channels= imgChannels,out_channels= 2,kernel_size= (200,10))
        self.encConv2 = nn.Conv2d(in_channels= 2,out_channels= 4,kernel_size= (200,10))
        self.encConv3 = nn.Conv2d(in_channels= 4,out_channels= 4,kernel_size= (200,10))
        self.encConv4 = nn.Conv2d(in_channels= 4,out_channels= 4,kernel_size= (200,10))
        self.encConv5 = nn.Conv2d(in_channels= 4,out_channels= 4,kernel_size= (200,10))

        self.flatten = nn.Flatten()

        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(4, 4, (200,10))
        self.decConv2 = nn.ConvTranspose2d(4, 4, (100,10))
        self.decConv3 = nn.ConvTranspose2d(4, 4, (100,10))
        self.decConv4 = nn.ConvTranspose2d(4, 2, (100,10))
        self.decConv5 = nn.ConvTranspose2d(2, imgChannels, (100,10))

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        x = F.relu(self.encConv5(x))
        print(x.shape)
        
        x = self.flatten(x)
        
        print("hi",x.shape)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view([128, 4, 606, 19])
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = F.relu(self.decConv4(x))
        x = torch.sigmoid(self.decConv5(x))
        return x

    def generate(self, batch_size=None, condition=None):
        #feed to decoder
        with torch.no_grad():
            return self.decoder(torch.randn(15656))

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
class M5(nn.Module):

    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        """Basic M5 model from the paper https://arxiv.org/pdf/1610.00087.pdf

        Args:
            n_input (int, optional): Number of inputs. Defaults to 1.
            n_output (int, optional): Number of outputs. Defaults to 35.
            stride (int, optional): Convolution stride. Defaults to 16.
            n_channel (int, optional): Output channels of convolution layers. Defaults to 32.
        Summary:
            =================================================================
            Layer (type:depth-idx)                   Param #
            =================================================================
            M5                                       --
            ├─Conv1d: 1-1                            5,635
            ├─BatchNorm1d: 1-2                       70
            ├─MaxPool1d: 1-3                         --
            ├─Conv1d: 1-4                            3,710
            ├─BatchNorm1d: 1-5                       70
            ├─MaxPool1d: 1-6                         --
            ├─Conv1d: 1-7                            7,420
            ├─BatchNorm1d: 1-8                       140
            ├─MaxPool1d: 1-9                         --
            ├─Conv1d: 1-10                           14,770
            ├─BatchNorm1d: 1-11                      140
            ├─MaxPool1d: 1-12                        --
            ├─Linear: 1-13                           2,485
            =================================================================
            Total params: 34,440
            Trainable params: 34,440
            Non-trainable params: 0
            =================================================================
        """
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze()
    
class Scattering_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        #Scattering hyperparameters
        T=16000
        J=12
        Q=10
        self.log_eps=1e-6
        #Layers
        self.scattering= Scattering1D(J=J,shape=T,Q=Q,T=1000)
        self.batchnorm=nn.BatchNorm1d(673)
        self.fc1= nn.Linear(673,300)
        self.fc2= nn.Linear(300,90)
        self.fc3= nn.Linear(90,35)

    def forward(self,x):
        #print(x.shape)
        x=self.scattering(x.squeeze())
        #print(x.shape)
        x=x[:,1:,:]
        x=torch.log(torch.abs(x)+self.log_eps)
        #print(x.shape)
        x=torch.mean(x,dim=-1)
        x=self.batchnorm(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return F.log_softmax(x,dim=-1)
    
    
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EncDecBaseModel(num_mels= 64, final_filter = 128, num_classes=35,input_length=1601)
    summary(model=model,device=device)
    model.to(device=device)

    test_input = torch.rand([256, 98, 64])
    test_input= test_input.to(device=device)
    label=torch.Tensor(98)
    label=label.to(device)
    
    test_output = model(test_input)
    
    print(test_output.size())