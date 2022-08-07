from args import get_model_args
from torch import nn
from layers import DecoderLayers, EncoderLayers
from torch import Tensor


class Model(nn.Module):
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

    def predict(self, dec_inp: Tensor, enc_inp: Tensor):
        if dec_inp.shape[0] == 1:
            enc_inp = self.encoder(x=enc_inp, mask=None)
        out, att = self.decoder(x=dec_inp, mask=None)
        out = self.fc(out)
        return enc_inp, nn.functional.log_softmax(out, dim=-1), att


def get_model(args, rank: int, voc_size: int) -> nn.Module:
    return Model(
        **get_model_args(args, voc_size, rank)
    )
