import torch
import torch.nn as nn
from .utils import EncoderBlock, ConvBlock, DecoderBlock, Downsampler
from torchinfo import summary

class Generator(nn.Module):
    def __init__(self, inp=13, oup=12, activation='leaky_relu'):
        super().__init__()
        
        self.enc0 = EncoderBlock(inp, 32, activation)
        self.enc1 = EncoderBlock(32, 64, activation)
        self.enc2 = EncoderBlock(64, 128, activation)
        self.enc3 = EncoderBlock(128, 256, activation)
        self.center = ConvBlock(256, 1024, activation)
        self.dec3 = DecoderBlock(1024, 256, activation, concat=256)
        self.dec2 = DecoderBlock(256, 128, activation, concat=128)
        self.dec1 = DecoderBlock(128, 64, activation, concat=64)
        self.dec0 = DecoderBlock(64, 32, activation, concat=32)
        self.conv = nn.Conv2d(32, oup, 1, padding='same')
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        enc_pool0, enc0 = self.enc0(x)
        enc_pool1, enc1 = self.enc1(enc_pool0)
        enc_pool2, enc2 = self.enc2(enc_pool1)
        enc_pool3, enc3 = self.enc3(enc_pool2)
        
        cen = self.center(enc_pool3)
        
        out = self.dec3(cen, enc3)
        out = self.dec2(out, enc2)
        out = self.dec1(out, enc1)
        out = self.dec0(out, enc0)
        
        out = self.conv(out)
        
        out = out.permute(0, 2, 3, 1)
        return out

# discriminator and downsample code inspired from :
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
# discriminator is a patchGAN - basically a convnet with B X H x W x C output
# instead of a single true/false output
class Discriminator(nn.Module):
    def __init__(self, inp=13, oup=12, activation='leaky_relu'):
        super().__init__()
        
        self.down1 = Downsampler(inp + oup, 64, 4, 2, apply_batchnorm=False, paddings=1, activation=activation)
        self.down2 = Downsampler(64, 128, 4, 2, paddings=1, activation=activation)
        self.down3 = Downsampler(128, 256, 4, 2, paddings=1, activation=activation)
        
        self.pad = nn.ZeroPad2d(1)
        
        self.down4 = Downsampler(256, 512, 4, 1, paddings=0, activation=activation)
        
        self.conv = nn.Conv2d(512, 1, 4)
        nn.init.normal_(self.conv.weight, 0., 0.02)
        
                
    def forward(self, x, y):
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        inp = torch.cat((x, y), 1)

        down1 = self.down1(inp)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        pad = self.pad(down3)
        
        down4 = self.down4(pad)
        new_pad = self.pad(down4)
        
        out = self.conv(new_pad)
        
        out = out.permute(0, 2, 3, 1)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(13, 12).to(device)
    disc = Discriminator(13, 12).to(device)
    
    x = torch.rand(2, 384, 384, 13).to(device)
    y = torch.rand(2, 384, 384, 12).to(device)
    with torch.no_grad():
        y = gen(x)
        print(y)
        print(y.shape)
        z = disc(x, y)
        print(z)
        print(z.shape)
    summary(gen)
    summary(disc)