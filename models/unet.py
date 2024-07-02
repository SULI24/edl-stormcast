# Based on UNet implementation here :
# https://colab.research.google.com/github/MarkDaoust/models/blob/segmentation_blogpost/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb
import torch
import torch.nn as nn
from .utils import EncoderBlock, ConvBlock, DecoderBlock, get_activation
from torchinfo import summary

class UNet(nn.Module):
    def __init__(self, input_shape=[13, 384, 384, 1], target_shape=[12, 384, 384, 1], enc_nodes= [32, 64, 128, 256], center=1024, dec_nodes=[256, 128, 64, 32], activation='relu', edl=False, edl_act = 'relu'):
        super().__init__()
        
        assert len(input_shape) == len(target_shape)
        assert len(enc_nodes) == len(dec_nodes)
        self.edl = edl
        self.target_shape = target_shape.copy()
        if self.edl:
            target_shape[0] *= 4
            
        
        self.enc_nodes = enc_nodes
        self.dec_nodes = dec_nodes
        self.enc_nodes.insert(0, input_shape[0])
        self.dec_nodes.insert(0, center)
                
        self.encs = nn.ModuleList([EncoderBlock(self.enc_nodes[i], self.enc_nodes[i + 1], activation=activation) for i in range(len(self.enc_nodes) - 1)])        
                             
        self.center = ConvBlock(enc_nodes[-1], center, activation)
        
        self.decs = nn.ModuleList([DecoderBlock(self.dec_nodes[i], self.dec_nodes[i + 1], activation=activation, concat=self.dec_nodes[i+1]) for i in range(len(self.dec_nodes) - 1)]) 
                             
        self.conv = nn.Conv2d(self.dec_nodes[-1], target_shape[0], 1, padding='same')
        if self.edl:
            self.edl_act = get_activation(edl_act)
            
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
        
        if self.edl:
            out = out.reshape(out.size(0), self.target_shape[0], self.target_shape[1], self.target_shape[2], 4)
            mu, logv, logalpha, logbeta = torch.split(out, 1, -1)
            v = self.edl_act(logv)
            alpha = self.edl_act(logalpha) + 1
            beta = self.edl_act(logbeta)
            out = torch.cat((mu, v, alpha, beta), dim=-1)
        else:
            out = out.unsqueeze(-1)
        
        return out
    
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(edl=True).to(device)
    x = torch.rand(2, 13, 384, 384, 1).to(device)
    with torch.no_grad():
        y = model(x)
        print(y)
        print(y.shape)
    summary(model)
