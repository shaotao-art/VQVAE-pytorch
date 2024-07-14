import torch
from torch import nn
from typing import Dict
from einops import rearrange

from models import Encoder, Decoder 



class VQVAE(nn.Module):
    def __init__(self, 
                 encoder_config: Dict,
                 codebook_size: int,
                 laten_dim: int,
                 decoder_config: Dict,
                 codebook_loss_w: float,
                 laten_loss_w: float,
                 ) -> None:
        super().__init__()
        self.codebook_loss_w = codebook_loss_w
        self.laten_loss_w = laten_loss_w
        
        self.encoder = Encoder(**encoder_config)
        self.codebook = nn.Embedding(num_embeddings=codebook_size, 
                                     embedding_dim=laten_dim)
        self.quant_conv = nn.Conv2d(encoder_config['channels_lst'][-1], 
                                    laten_dim, 1, 1, 0)

        self.post_quant_conv = nn.Conv2d(laten_dim, 
                                         decoder_config['channels_lst'][0], 1, 1, 0)
        self.decoder = Decoder(**decoder_config)
        
        
        self.reconstruct_loss_fn = nn.MSELoss()
        self.codebook_loss_fn = nn.MSELoss()
        self.encoder_laten_loss_fn = nn.MSELoss()
        
    def train_loss(self, img):
        enc_out = self.encoder(img)
        laten = self.quant_conv(enc_out)
        
        quant_out = self.quant_laten(laten)
        # make quanted_laten gradient copyed to laten in backward
        quant_laten = laten + (quant_out['quant_laten'] - laten).detach()

        dec_inp = self.post_quant_conv(quant_laten)
        reconstruct_img = self.decoder(dec_inp)
        
        # backward through encoder and decoder
        reconst_loss = self.reconstruct_loss_fn(reconstruct_img, img)
        # backward through codebook to make codebooks vector more close to encoder's output
        codebook_loss = self.codebook_loss_fn(self.codebook(quant_out['closest_idx_2d']).permute(0, 3, 1, 2), 
                                              laten.detach())
        # backward through ecnoder to make encoder's output more close to laten
        laten_loss = self.encoder_laten_loss_fn(laten, quant_laten.detach())
        loss = reconst_loss + codebook_loss * self.codebook_loss_w + laten_loss * self.laten_loss_w
        return dict(loss=loss, 
                    reconst_loss=reconst_loss, 
                    codebook_loss=codebook_loss,
                    laten_loss=laten_loss)
        
    @torch.no_grad()
    def encode_img(self, img):
        self.eval()
        enc_out = self.encoder(img)
        laten = self.quant_conv(enc_out)
        quant_out = self.quant_laten(laten)
        return quant_out
    
    @torch.no_grad()
    def decode_laten(self, laten):
        self.eval()
        dec_inp = self.post_quant_conv(laten)
        reconstruct_img = self.decoder(dec_inp)
        return reconstruct_img

    @torch.no_grad()
    def enc_and_dec_img(self, img):
        self.eval()
        return self.decode_laten(self.encode_img(img)['quant_laten'])
    
    @torch.no_grad()
    def quant_laten(self, laten):
        b_s, _, laten_h, laten_w = laten.shape
        # (b, laten_dim, h, w) --> (b * h * w, laten_dim)
        laten_reshaped = rearrange(laten, 'b c h w -> (b h w) c')
        # (b*h*w, 1, laten_dim) - (codebook_size, laten_dim) --> (b*h*w, codebook_size, laten_dim
        a_minus_b = laten_reshaped.unsqueeze(1) - self.codebook.weight # (b*h*w, codebook_size, laten_dim)
        # use L2 distance
        l2_distance = torch.mean(a_minus_b ** 2, dim=-1) # (b*h*w, codebook_size)
        closest_idx_flatten = l2_distance.argmin(dim=-1) # (b*h*w, )
        closest_idx_2d = closest_idx_flatten.reshape(b_s, laten_h, laten_w) # (b, h, w)
        quant_laten = self.codebook(closest_idx_2d).permute(0, 3, 1, 2) # (b, laten_dim, h, w)
        return dict(closest_idx_2d=closest_idx_2d, quant_laten=quant_laten)