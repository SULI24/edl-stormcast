import torch
import torch.nn as nn

def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
        

class Downsampler(nn.Module):
    def __init__(self, inp, oup, size, stride, apply_batchnorm=True, paddings=0, activation='leaky_relu'):
        super().__init__()
        self.bn_do = apply_batchnorm
        self.conv = nn.Conv2d(inp, oup, size, stride=stride, padding=paddings, bias=False)
        nn.init.normal_(self.conv.weight, 0., 0.02)
        if self.bn_do:
            self.bn = nn.BatchNorm2d(oup)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        if self.bn_do:
            out = self.bn(out)
        return self.lrelu(out)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, activation='relu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, padding='same'),
            nn.BatchNorm2d(oup),
            get_activation(activation),
            nn.Conv2d(oup, oup, 3, padding='same'),
            nn.BatchNorm2d(oup),
            get_activation(activation)
        )

    def forward(self, x):
        return self.conv(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, inp, oup, activation='relu', resnet_style=False):
        super().__init__()
        self.enc = ConvBlock(inp, oup, activation)
        self.resnet_style = resnet_style
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        out = self.enc(x)
        if self.resnet_style:
            out = torch.cat([out, x], dim=-1)
        return self.pool(out), out
    
class DecoderBlock(nn.Module):
    def __init__(self, inp, oup, activation='relu', resnet_style=False, concat=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(inp, oup, 2, stride=2)
        self.dec = nn.Sequential(
            nn.BatchNorm2d(oup + concat),
            get_activation(activation),
            nn.Conv2d(oup + concat, oup, 3, padding='same'),
            nn.BatchNorm2d(oup),
            get_activation(activation),
            nn.Conv2d(oup, oup, 3, padding='same'),
            nn.BatchNorm2d(oup),
            get_activation(activation)
        )
        
    def forward(self, x, concat=None):
        out = self.conv(x)
        if concat is not None:
            out = torch.cat([concat, out], dim=1)
        return self.dec(out)