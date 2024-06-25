# Based on UNet implementation here :
# https://colab.research.google.com/github/MarkDaoust/models/blob/segmentation_blogpost/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb
import torch
import torch.nn as nn
from .utils import EncoderBlock, ConvBlock, DecoderBlock
from torchinfo import summary

class UNet(nn.Module):
    def __init__(self, inp=13, oup=12, activation='relu'):
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
    
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(13, 12).to(device)
    x = torch.rand(2, 384, 384, 13).to(device)
    with torch.no_grad():
        y = model(x)
        print(y)
        print(y.shape)
    summary(model)
