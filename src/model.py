import torch
import torch.nn as nn
import torch.nn.functional as F

class M5(nn.Module):

    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        """Basic M5 model from the paper https://arxiv.org/pdf/1610.00087.pdf (Only for testing)

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