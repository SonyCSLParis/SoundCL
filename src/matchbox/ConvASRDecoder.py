# Modified code : original was taken from the nvidia nemo library https://github.com/NVIDIA/NeMo/blob/557c4b7ae766faf050374e6b9a862e2e67385b10/nemo/collections/asr/modules/conv_asr.py#L56

from matchbox.ConvBlock import init_weights
from typing import Optional
import torch
import torch.nn as nn

class ConvASRDecoderClassification(nn.Module):
    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        init_mode: Optional[str] = "xavier_uniform",
        return_logits: bool = True,
        pooling_type='avg',
    ):
        """Matchbox net decoder

        Args:
            feat_in (int): number of futures inputed
            num_classes (int): number of classes on the output layer
            init_mode (Optional[str], optional): Weight initialisation strategy. Defaults to "xavier_uniform".
            return_logits (bool, optional): Return logit or softmax for greater flexibility. Defaults to True.
            pooling_type (str, optional): Pooling layer type on the output of the encoder. Defaults to 'avg'.

        Raises:
            ValueError: if pooling type isn't supported
        """
        super(ConvASRDecoderClassification, self).__init__()

        self._feat_in = feat_in
        self._return_logits = return_logits
        self._num_classes = num_classes

        if pooling_type == 'avg':
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max':
            self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError('Pooling type chosen is not valid. Must be either `avg` or `max`')

        self.decoder_layers = torch.nn.Sequential(torch.nn.Linear(self._feat_in, self._num_classes, bias=True))
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        batch, in_channels, timesteps = encoder_output.size()

        encoder_output = self.pooling(encoder_output).view(batch, in_channels)  # [B, C]
        logits = self.decoder_layers(encoder_output)  # [B, num_classes]

        if self._return_logits:
            return logits

        return torch.nn.functional.softmax(logits, dim=-1)