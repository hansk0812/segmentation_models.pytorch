"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
import torch
from torch import nn

from ._base import EncoderMixin

class Resnet3d(nn.Module, EncoderMixin):
    def __init__(self, in_channels=1, out_channels=[24,24,48,96,192]):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._depth = len(out_channels)
        self.net = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True)
        self.net = nn.ModuleList(self.net.blocks[:-1])
        
        assert in_channels <= 3, "pretrained model only has 3 channel inputs"
        first_conv = self.net[0]._modules['conv']._modules['conv_t']
        first_conv._parameters['weight'] = first_conv._parameters['weight'][:,0:in_channels,...]

    def forward(self, x):
        output = [x]
        for module in self.net:
            x = module(x)
            output.append(x)
        return output

