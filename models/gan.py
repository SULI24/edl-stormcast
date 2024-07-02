import torch
import torch.nn as nn
from .utils import EncoderBlock, ConvBlock, DecoderBlock, Downsampler
from torchinfo import summary

class Generator(nn.Module):
    def __init__(self, input_shape=[13, 384, 384, 1], target_shape=[12, 384, 384, 1], enc_nodes= [32, 64, 128, 256], center=1024, dec_nodes=[256, 128, 64, 32], activation='leaky_relu'):
        super().__init__()
        
        assert len(input_shape) == len(target_shape)
        assert len(enc_nodes) == len(dec_nodes)
        
        self.enc_nodes = enc_nodes
        self.dec_nodes = dec_nodes
        self.enc_nodes.insert(0, input_shape[0])
        self.dec_nodes.insert(0, center)
                
        self.encs = nn.ModuleList([EncoderBlock(self.enc_nodes[i], self.enc_nodes[i + 1], activation=activation) for i in range(len(self.enc_nodes) - 1)])        
                             
        self.center = ConvBlock(enc_nodes[-1], center, activation)
        
        self.decs = nn.ModuleList([DecoderBlock(self.dec_nodes[i], self.dec_nodes[i + 1], activation=activation, concat=self.dec_nodes[i+1]) for i in range(len(self.dec_nodes) - 1)]) 
                             
        self.conv = nn.Conv2d(self.dec_nodes[-1], target_shape[0], 1, padding='same')
        
    def forward(self, x):
        x = x.squeeze(-1)
        pool = x
        encs = []
        for i, enc_n in enumerate(self.encs):
            pool, enc_i = enc_n(pool)
            encs.insert(0, enc_i)
        
        out = self.center(pool)
        
        for i, dec_n in enumerate(self.decs):
            out = dec_n(out, encs[i])
        
        out = self.conv(out)
        out = out.unsqueeze(-1)
        
        return out

# discriminator and downsample code inspired from :
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
# discriminator is a patchGAN - basically a convnet with B X H x W x C output
# instead of a single true/false output
class Discriminator(nn.Module):
    def __init__(self, input_shape=[13, 384, 384, 1], target_shape=[12, 384, 384, 1], downsample_nodes= [64, 128, 256, 512], batchnorms=[False, True, True, True], paddings=[1,1,1,0], strides=[2, 2, 2, 1, 1], activation='leaky_relu'):
        super().__init__()
        
        self.downsample_nodes = downsample_nodes
        self.downsample_nodes.insert(0, input_shape[0] + target_shape[0])
        
        self.downs = nn.ModuleList([Downsampler(self.downsample_nodes[i], self.downsample_nodes[i + 1], 4, strides[i], paddings=paddings[i], apply_batchnorm=batchnorms[i], activation=activation) for i in range(len(self.downsample_nodes) - 1)])
                
        self.pad = nn.ZeroPad2d(1)
                
        self.conv = nn.Conv2d(self.downsample_nodes[-1], strides[-1], 4)
        nn.init.normal_(self.conv.weight, 0., 0.02)
        
                
    def forward(self, x, y):
        out = torch.cat((x, y), 1)
        out = out.squeeze(-1)
        for i, down in enumerate(self.downs[:-1]):
            out = down(out)
        out = self.pad(out)
        out = self.downs[-1](out)
        out = self.pad(out)
        out = self.conv(out)
        out = out.unsqueeze(-1)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    
    x = torch.rand(2, 13, 384, 384, 1).to(device)
    y = torch.rand(2, 12, 384, 384, 1).to(device)
    with torch.no_grad():
        y = gen(x)
        print(y)
        print(y.shape)
        z = disc(x, y)
        print(z)
        print(z.shape)
    summary(gen)
    summary(disc)