from operator import mod
from torch import nn
from layers import DecoderLayers, EncoderLayers
from torch import Tensor
import torch


class Model(nn.Module):
    def __init__(
            self,
            enc_parms: dict,
            dec_params: dict,
            h: int,
            d_model: int,
            device: str
            ) -> None:
        super().__init__()
        self.encoder = EncoderLayers(
            d_model=d_model,
            h=h,
            device=device,
            **enc_parms
        )
        self.decoder = DecoderLayers(
            d_model=d_model,
            h=h,
            device=device,
            **dec_params
        )

    def forward(
            self, 
            enc_inp: Tensor, 
            dec_inp: Tensor, 
            enc_mask: Tensor, 
            dec_mask: Tensor
            ):
        enc_vals = self.encoder(x=enc_inp, mask=enc_mask)
        out, att = self.decoder(
            x=dec_inp, mask=dec_mask, enc_values=enc_vals
            )
        return torch.softmax(out, dim=-1), att


if __name__ == '__main__':
    params = {
        'd_model': 512,
        'h': 8,
        'device': 'cuda'
    }

    enc_params = {
        'n_layers': 4,
        'voc_size': 20,
        'hidden_size': 256,
        'p_dropout': 0.1
    }

    dec_params = {
        'voc_size': 30,
        'n_layers': 4,
        'p_dropout': 0.1,
        'hidden_size': 256
    }

    model = Model(
        enc_parms=enc_params,
        dec_params=dec_params,
        **params
    ).cuda()

    enc_inp = torch.LongTensor([[2,5,4], [3,2, 0]]).cuda()
    enc_mask = torch.BoolTensor([[0, 0, 0], [0, 0, 1]]).cuda()
    dec_inp = torch.LongTensor([[2,5,4,7,5,0,0], [3, 2, 8, 8, 9, 9, 8]]).cuda()
    dec_mask = torch.BoolTensor([[0, 0, 0, 0, 0, 1,1], [0, 0, 0,0,0,0,0]]).cuda()

    out, att = model(
        enc_inp=enc_inp,
        dec_inp=dec_inp,
        enc_mask=enc_mask,
        dec_mask=dec_mask
    )