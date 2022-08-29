from typing import Tuple, Union
from args import get_model_args
from interfaces import IPredictorStep
from torch import nn
from layers import (
    Attention,
    DecoderLayers,
    EncoderLayers,
    PackedGRU,
    RNNDecoder,
    RNNEncoder
    )
from torch import Tensor
import torch


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


class Seq2SeqRNN(nn.Module):
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
        enc_lengths = self.get_lengths(enc_mask).cpu()
        dec_lengths = self.get_lengths(dec_mask).cpu()
        enc_values, h = self.encoder(enc_inp, enc_lengths)
        result, attention = self.decoder(
            enc_values=enc_values,
            hn=h,
            x=dec_inp,
            lengths=dec_lengths
            )
        return nn.functional.log_softmax(result, dim=-1), attention

    def predict(
            self,
            dec_inp: Tensor,
            enc_inp: Tensor,
            enc_mask,
            h: Tensor,
            key=None,
            value=None
            ):
        enc_lengths = self.get_lengths(enc_mask)
        if key is None and value is None:
            enc_values, h = self.encoder(enc_inp, enc_lengths)
        else:
            enc_values = None
        h, att, result, key, value = self.decoder.predict(
            hn=h,
            x=dec_inp,
            enc_values=enc_values,
            key=key,
            value=value
        )
        return h, att, result, key, value


class Seq2SeqBasic(nn.Module):
    def __init__(
            self,
            voc_size: int,
            emb_size: int,
            n_layers: int,
            hidden_size: int,
            padding_idx: int,
            padding_value: int,
            p_dropout: float,
            max_len: int,
            *args,
            **kwargs
            ) -> None:
        super().__init__()
        self.enc_embedding = nn.Embedding(
            voc_size,
            emb_size,
            padding_idx=padding_idx
        )
        self.dec_embedding = nn.Embedding(
            voc_size,
            emb_size,
            padding_idx=padding_idx
        )
        self.encoder = PackedGRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers,
            padding_value=padding_value
        )
        self.decoder = PackedGRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers,
            padding_value=padding_value
        )
        self.attention = Attention(hidden_size)
        self.pred_net = nn.Linear(
            hidden_size,
            voc_size
        )
        self.padding_value = padding_value
        self.max_len = max_len

    def get_lengths(self, mask: Tensor) -> Tensor:
        return (~mask).sum(dim=-1)

    def forward(
            self,
            enc_inp: Tensor,
            dec_inp: Tensor,
            enc_mask: Tensor,
            dec_mask: Tensor
            ):
        enc_lengths = self.get_lengths(enc_mask).cpu()
        dec_lengths = self.get_lengths(dec_mask).cpu()
        enc_values, hn = self.get_enc_values(enc_inp, length=enc_lengths)
        max_len = dec_lengths.max().item()
        attention = list()
        result = list()
        for i in range(max_len):
            inp = dec_inp[:, i:i+1]
            preds, hn, att = self.predict(
                enc_values=enc_values,
                dec_inp=inp,
                hn=hn
                )
            result.append(preds)
            attention.append(att[:, -1:, :])
        result = torch.hstack(result)
        attention = torch.hstack(attention)
        return nn.functional.log_softmax(result, dim=-1), attention

    def predict(
            self,
            enc_values: Tensor,
            dec_inp: Tensor,
            hn: Tensor
            ):
        hn, att = self.attention(
            query=hn, key=enc_values, value=enc_values
            )
        dec_inp = self.dec_embedding(dec_inp)
        output, hn = self.decoder(
            x=dec_inp, lengths=torch.ones(dec_inp.shape[0]), hn=hn
        )
        preds = self.pred_net(output)
        return preds, hn, att

    def get_enc_values(
            self,
            enc_inp: Tensor,
            length: Union[Tensor, None] = None
            ):
        if length is None:
            length = torch.ones(enc_inp.shape[0]) * enc_inp.shape[1]
        enc_inp = self.enc_embedding(enc_inp)
        enc_values, hn = self.encoder(enc_inp)
        return enc_values, hn


def get_model(
        args,
        rank: int,
        voc_size: int,
        pad_idx: int
        ) -> nn.Module:
    if args.model == 'seq2seqrnn':
        return Seq2SeqRNN(
            **get_model_args(args, voc_size, rank, pad_idx)
        )
    if args.model == 'basicrnn':
        return Seq2SeqBasic(
            **get_model_args(args, voc_size, rank, pad_idx)
        )
    return Transformer(
        **get_model_args(args, voc_size, rank, pad_idx)
        )
