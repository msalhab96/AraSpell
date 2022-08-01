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
            device: str,
            voc_size: int
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
        self.fc = nn.Linear(d_model, voc_size)
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
        out = self.fc(out)
        return torch.softmax(out, dim=-1), att


def get_model(rank, voc_size):
    params = {
        'd_model': 256,
        'h': 8,
        'device': f'cuda:{rank}',
        'voc_size': voc_size
    }

    enc_params = {
        'n_layers': 5,
        'voc_size': voc_size,
        'hidden_size': 256,
        'p_dropout': 0.1
    }

    dec_params = {
        'n_layers': 5,
        'p_dropout': 0.1,
        'hidden_size': 256,
        'voc_size': voc_size,
    }

    model = Model(
        enc_parms=enc_params,
        dec_params=dec_params,
        **params
    )
    return model

    # enc_inp = torch.LongTensor([[2,5,4], [3,2, 0]]).cuda()
    # enc_mask = torch.BoolTensor([[0, 0, 0], [0, 0, 1]]).cuda()
    # dec_inp = torch.LongTensor([[2,5,4,7,5,0,0], [3, 2, 8, 8, 9, 9, 8]]).cuda()
    # dec_mask = torch.BoolTensor([[0, 0, 0, 0, 0, 1,1], [0, 0, 0,0,0,0,0]]).cuda()

    # out, att = model(
    #     enc_inp=enc_inp,
    #     dec_inp=dec_inp,
    #     enc_mask=enc_mask,
    #     dec_mask=dec_mask
    # )