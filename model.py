from typing import Tuple
from args import get_model_args
from torch import nn
from layers import DecoderLayers, EncoderLayers, RNNDecoder, RNNEncoder
from torch import Tensor


class Transformer(nn.Module):
    def __init__(
            self,
            enc_params: dict,
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
            **enc_params
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
            x=dec_inp, mask=dec_mask, enc_values=enc_vals, key_mask=enc_mask
            )
        out = self.fc(out)
        return nn.functional.log_softmax(out, dim=-1), att

    def predict(
            self,
            dec_inp: Tensor,
            enc_inp: Tensor,
            enc_mask=None,
            dec_mask=None,
            *args,
            **kwargs
            ):
        if dec_inp.shape[1] == 1:
            enc_inp = self.encoder(x=enc_inp, mask=enc_mask)
        out, att = self.decoder(
            x=dec_inp, mask=dec_mask, enc_values=enc_inp, key_mask=enc_mask
            )
        out = self.fc(out)
        return enc_inp, nn.functional.log_softmax(out, dim=-1), att


class RNN(nn.Module):
    def __init__(
            self,
            voc_size: int,
            emb_size: int,
            n_layers: int,
            hidden_size: int,
            bidirectional: bool,
            padding_idx: int,
            padding_value: int,
            p_dropout: float,
            max_len: int
            ) -> None:
        super().__init__()
        self.encoder = RNNEncoder(
            voc_size=voc_size,
            emb_size=emb_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=bidirectional,
            padding_idx=padding_idx,
            padding_value=padding_value
        )
        self.decoder = RNNDecoder(
            max_len=max_len,
            voc_size=voc_size,
            emb_size=emb_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=bidirectional,
            padding_idx=padding_idx,
            padding_value=padding_value
        )

    def get_lengths(self, mask: Tensor) -> Tensor:
        return (~mask).sum(dim=-1)

    def forward(
            self,
            enc_inp: Tensor,
            dec_inp: Tensor,
            enc_mask: Tensor,
            dec_mask: Tensor
            ) -> Tuple[Tensor, Tensor]:
        enc_lengths = self.get_lengths(enc_mask)
        dec_lengths = self.get_lengths(dec_mask)
        enc_values, h = self.encoder(enc_inp, enc_lengths)
        result, attention = self.decoder(
            enc_values=enc_values,
            hn=h,
            x=dec_inp,
            lengths=dec_lengths
            )
        return result, attention


def get_model(
        args,
        rank: int,
        voc_size: int,
        pad_idx: int
        ) -> nn.Module:
    if args.model == 'rnn':
        return RNN(
            **get_model_args(args, voc_size, rank, pad_idx)
        )
    return Transformer(
        **get_model_args(args, voc_size, rank, pad_idx)
        )
