from torch import nn
from layers import *

class Encoder(nn.Module):
    def __init__(self, 
                 in_channel, 
                 init_channels,
                 channels_lst,
                 num_groups_lst,
                 num_res_layers_per_resolution,
                 layer_with_attention,
                 save_hidden=False,
                 d_time_embed=None) -> None:
        super().__init__()
        self.save_hidden = save_hidden

        self.init_conv = nn.Conv2d(in_channel, init_channels, 3, 1, 1)
        channels = [init_channels] + channels_lst

        self.encoder_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(channels, channels[1:])):
            layers = nn.Module()
            
            # res layers
            res_layers = nn.ModuleList()
            for _ in range(num_res_layers_per_resolution):
                res_layers.append(ResBlock(in_channel=in_dim,
                                    out_channel=out_dim,
                                    num_groups=num_groups_lst[layer_idx],
                                    d_time_embed=d_time_embed
                                    ))
                in_dim = out_dim
            
            # attention
            if layer_with_attention[layer_idx] == True:
                res_layers.append(AttentionC(out_dim,
                                        num_groups=num_groups_lst[layer_idx]))
            layers.block = res_layers
            
            
            # if is the last block, do not add downsample
            is_last = (layer_idx == len(channels_lst) - 1)
            if not is_last:
                layers.down_sample = Downsample(out_dim)
            else:
                layers.down_sample = nn.Identity()
            self.encoder_layers.append(layers)
    
    def forward(self, x, time_embed=None):
        x = self.init_conv(x)
        hiddens = None
        if self.save_hidden:
            hiddens = []
        for layer in self.encoder_layers:
            for b in layer.block:
                if isinstance(b, ResBlock):
                    x = b(x, time_embed)
                else:
                    x = b(x)
            if self.save_hidden:
                hiddens.append(x)
            
            x = layer.down_sample(x)
        if self.save_hidden:
            return dict(enc_out=x, hiddens=hiddens)
        else:
            return x
        
        
        
class Decoder(nn.Module):
    def __init__(self,
                 out_channel,
                 channels_lst,
                 num_groups_lst,
                 num_res_layers_per_resolution,
                 layer_with_attention,
                 d_time_embed=None,
                 use_concat=False) -> None:
        super().__init__()
        channels = [channels_lst[0]] + channels_lst
        
        self.decoder_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(channels, channels[1:])):
            layers = nn.Module()
            
            # res layers and attention
            res_layers = nn.ModuleList()
            if use_concat:
                in_dim = in_dim + out_dim
            for _ in range(num_res_layers_per_resolution):
                res_layers.append(ResBlock(in_channel=in_dim,
                                    out_channel=out_dim,
                                    num_groups=num_groups_lst[layer_idx],
                                    d_time_embed=d_time_embed
                                    ))
                in_dim = out_dim
            if layer_with_attention[layer_idx] == True:
                res_layers.append(AttentionC(out_dim,
                                        num_groups=num_groups_lst[layer_idx]))
            layers.block = res_layers
            
            # up sample
            is_last = (layer_idx == len(channels_lst) - 1)
            if not is_last:
                layers.up_sample = Upsample(out_dim)
            else:
                layers.up_sample = nn.Identity()
            self.decoder_layers.append(layers)
        
        self.conv_out = nn.Sequential(
                ResBlock(out_dim, out_dim, num_groups_lst[-1], d_time_embed),
                nn.Conv2d(out_dim, out_channel, 3, 1, 1)
            )
    
        
    def forward(self, x, time_embed=None, hiddens=None):
        for layer in self.decoder_layers:
            if hiddens is not None:
                x = torch.cat([x, hiddens.pop()], dim=1)
            for b in layer.block:
                if isinstance(b, ResBlock):
                    x = b(x, time_embed)
                else:
                    x = b(x)
            x = layer.up_sample(x)
        return self.conv_out(x)
    
    
class UNet(nn.Module):
    def __init__(self,
                in_channel = 3,
                out_channel = 3,
                init_channels = 64,
                channels_lst = [64, 128, 256, 256],
                num_groups_lst = [32, 32, 32, 32],
                num_res_layers_per_resolution = 2,
                layer_with_attention = [False, False, False, True],
                d_time_embed=None) -> None:
        super().__init__()
        if d_time_embed is not None:
            self.time_mlp = nn.Sequential(
                TimeEmbed(d_embed=d_time_embed),
                nn.Linear(d_time_embed, d_time_embed),
                nn.SiLU(),
                nn.Linear(d_time_embed, d_time_embed)
            )
        else:
            self.time_mlp = None
            
            
        self.encoder = Encoder(in_channel = in_channel,
                                init_channels = init_channels,
                                channels_lst = channels_lst,
                                num_groups_lst = num_groups_lst,
                                num_res_layers_per_resolution = num_res_layers_per_resolution,
                                layer_with_attention = layer_with_attention,
                                save_hidden=True,
                                d_time_embed=d_time_embed
                            )
        self.middle = UnetMiddleBlock(in_channel=channels_lst[-1],
                                      num_groups=num_groups_lst[-1])
        
        self.decoder = Decoder(
                out_channel = out_channel,
                channels_lst = list(reversed(channels_lst)),
                num_groups_lst = list(reversed(num_groups_lst)),
                num_res_layers_per_resolution = list(reversed(num_res_layers_per_resolution)),
                layer_with_attention = list(reversed(layer_with_attention)),
                d_time_embed=d_time_embed,
                use_concat=True
        )
    def forward(self, x, time):
        if self.time_mlp is not None:
            time_embed = self.time_mlp(time)
        else:
            time_embed = None
        enc_out = self.encoder(x, time_embed=time_embed)
        middle_out = self.middle(enc_out['enc_out'])
        return self.decoder(middle_out, hiddens=enc_out['hiddens'], time_embed=time_embed)
        