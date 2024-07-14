import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
import math

        
class Block(nn.Module):
    def __init__(self, 
                 in_channel, 
                 out_channel,
                 num_groups,
                 ) -> None:
        super().__init__()
        assert in_channel % num_groups == 0
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=3,
                              padding=1,
                              stride=1)
        self.norm = nn.GroupNorm(num_channels=out_channel,
                                 num_groups=num_groups)
        self.act = nn.SiLU()

    
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, 
                 in_channel,
                 out_channel, 
                 num_groups,
                 d_time_embed=None,
                 ) -> None:
        super().__init__()
        self.block1 = Block(in_channel, out_channel, num_groups)
        self.block2 = Block(out_channel, out_channel, num_groups)
        
        self.skip_connection = nn.Conv2d(in_channel, out_channel, 1, 1, 0) if in_channel != out_channel else nn.Identity()
        self.time_mlp = None
        if d_time_embed is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_time_embed, out_channel * 2),
            )
            
    def forward(self, x, time_embed=None):
        res = x 
        x = self.block1(x)
        if time_embed is not None:
            assert self.time_mlp is not None
            assert len(x.shape) == 4
            scale, bias = torch.chunk(self.time_mlp(time_embed), chunks=2, dim=-1)
            x = x * (scale[:, :, None, None] + 1) + bias[:, :, None, None]
        x = self.block2(x)
        return self.skip_connection(res) + x        
    
    
class Upsample(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        )
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2., mode='nearest')
        return self.layer(x)

class Downsample(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channel * 4, in_channel, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.layer(x)
    

class AttentionC(nn.Module):
    """attention in CNN, use C as the seq len"""
    def __init__(self, 
                 in_channel, 
                 num_groups,
                 drop_p=0.) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_channels=in_channel,
                                 num_groups=num_groups)
        self.to_qkv = nn.Conv2d(in_channel, in_channel * 3, 1, 1, 0)
        self.dropout = nn.Dropout(drop_p)
        
        self.conv_out = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        
    def forward(self, x):
        assert len(x.shape) == 4
        res = x
        x = self.norm(x)
        qkv = torch.chunk(self.to_qkv(x), chunks=3, dim=1)
        h, w = qkv[0].shape[-2:]
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b c (h w)'), qkv)
        scale = q.shape[-1] ** -0.5
        atten = einsum('b i d, b j d -> b i j', q, k) * scale
        atten = F.softmax(atten, dim=-1)
        atten = self.dropout(atten)

        out = einsum('b i j, b j d -> b i d', atten, v)
        out = rearrange(out, 'b i (h w) -> b i h w', h=h, w=w)
        out = self.conv_out(out)
        return out + res


class TimeEmbed(nn.Module):
    def __init__(self, d_embed, max_freq=10 ** 4) -> None:
        super().__init__()
        assert d_embed // 2 == d_embed / 2
        self.half_d = d_embed // 2
        self.max_freq = max_freq
        
    
    def forward(self, x):
        down = torch.exp(- math.log(self.max_freq) * (torch.arange(self.half_d).float() / (self.half_d - 1))).to(x.device)
        inner = einsum('i, j -> i j', x, down)
        sinx = torch.sin(inner)
        cosx = torch.cos(inner)
        out = torch.cat([sinx, cosx], dim=-1)
        return out
    
    
    
class UnetMiddleBlock(nn.Module):
    def __init__(self, 
                 in_channel, 
                 num_groups) -> None:
        super().__init__()
        self.middle_layers = nn.Sequential(
                ResBlock(in_channel=in_channel,
                        out_channel=in_channel,
                        num_groups=num_groups),
                AttentionC(in_channel=in_channel,
                           num_groups=num_groups),
                ResBlock(in_channel,
                        in_channel,
                        num_groups)
            )
        
    def forward(self, x):
        return self.middle_layers(x)
    
