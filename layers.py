import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union
from torch import Tensor
from utils import get_positionals
from torch.nn.utils.rnn import (
    pad_packed_sequence, pack_padded_sequence
    )


class MultiHeadAtt(nn.Module):
    """Implements the multi-head attention module

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        p_dropout (float): The dropout ratio.
        device (str): The device to map the operations to.
    """
    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        assert d_model % h == 0, 'd_model is not divisible by h'
        self.fc_key = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.fc_query = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.fc_value = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.proj_fc = nn.Linear(
            in_features=2 * d_model,
            out_features=d_model,
        )
        self.dropout = nn.Dropout(p_dropout)
        self.d_model = d_model
        self.h = h
        self.dk = d_model // h
        self.sqrt_dk = math.sqrt(self.dk)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def _get_scaled_att(
            self,
            Q: Tensor,
            K: Tensor,
            mask: Union[Tensor, None] = None,
            query_mask: Union[Tensor, None] = None,
            key_mask: Union[Tensor, None] = None
            ) -> Tensor:
        """Calculates the scaled attention map
        by calculating softmax(matmul(Q, K.T)/sqrt(dk))
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk]
            K (Tensor): The Key tensor of shape [h * B, dk, Tk]
            mask (Union[Tensor, None]): The mask tensor where its value is
            True when there's a padding in that position, of shape [B, M].
            Default None.
        Returns:
            Tensor: The scaled attention weights of shape
            [B * h, Tq, Tk]
        """
        result = torch.matmul(Q, K)
        result = result / self.sqrt_dk
        if mask is not None:
            # Used for self attention!
            mask = self.get_mask(Q, K, mask)
            result = result.masked_fill(mask, -1e9)
        if all([
                item is not None for item in [query_mask, key_mask]
                ]):
            mask = self.get_key_query_mask(query_mask, key_mask)
            result = result.masked_fill(mask, -1e9)
        return self.softmax(result)

    def get_key_query_mask(
            self, query_mask: Tensor, key_mask: Tensor
            ) -> Tensor:
        """Given the query and the key masks of shape [B, M], it returns
        the encoder decoder mask of shape [B * h, Tq, Tk].

        Args:
            query_mask (Tensor): The query mask of shape [B, Tq]
            key_mask (Tensor): The key mask of shape [B, Tk]

        Returns:
            Tensor: The encoder-decoder mask of shape [B, Tq, Tk].
        """
        batch_size, t_query = query_mask.shape
        # [B, h * Tq]
        mask = key_mask.repeat(1, self.h * t_query)
        # [B * h, Tq, Tk]
        mask = mask.reshape(batch_size * self.h,  t_query, -1)
        # [B, h * Tq]
        query_mask = query_mask.repeat(1, self.h)
        # [B * h, Tq, 1]
        query_mask = query_mask.view(self.h * batch_size, -1, 1)
        mask = mask | query_mask
        return mask

    def perform_att(
            self,
            Q: Tensor,
            K: Tensor,
            V: Tensor,
            mask: Union[Tensor, None] = None,
            query_mask: Union[Tensor, None] = None,
            key_mask: Union[Tensor, None] = None
            ) -> Tensor:
        """Performs multi-head scaled attention
        by calculating softmax(matmul(Q, K.T)/sqrt(dk)).V
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk].
            K (Tensor): The Key tensor of shape [h * B, dk, Tk].
            V (Tensor): The Value tensor of shape [h * B, Tk, dk].
            mask (Union[Tensor, None]): The mask tensor where its value is
            True when there's a padding in that position, of shape [B, M].
            Default None.
        Returns:
            Tuple[Tensor, Tensor]: The attention matrix of shape
            [B * h, Tq, Tk] and the scaled attention value of
            shape [B * h, Tq, dk].
        """
        att = self._get_scaled_att(
            Q, K, mask=mask, query_mask=query_mask, key_mask=key_mask
            )
        result = torch.matmul(att, V)
        return att, result

    def _reshape(self, *args) -> List[Tensor]:
        """Reshapes all given list of tensor
        from [B, T, N] to [B, T, h, dk]
        Returns:
            List[Tensor]: list of all reshaped tensors
        """
        return [
            item.contiguous().view(-1, item.shape[1], self.h, self.dk)
            for item in args
        ]

    def _pre_permute(self, *args) -> List[Tensor]:
        """Permutes all given list of tensors
        from [B, T, h, dk] to become [h, B, T, dk].

        Returns:
            List[Tensor]: List of all permuted tensors.
        """
        return [
            item.permute(2, 0, 1, 3)
            for item in args
        ]

    def _change_dim(self, *args) -> List[Tensor]:
        """Changes the dimensionality of all passed tensores
        from [B, T, N] to [B * h, T, dk]

        Returns:
            List[Tensor]: List of the modified tensors.
        """
        result = self._reshape(*args)  # [B, T, h, dk]
        result = self._pre_permute(*result)  # [h, B, T, dk]
        return [
            item.permute(1, 0, 2, 3).contiguous().view(
                -1, item.shape[2], item.shape[3]
                )
            for item in result
        ]

    def get_mask(
            self,
            query: Tensor,
            key: Tensor,
            mask: Union[None, Tensor],
            *args, **kwargs
            ) -> Tensor:
        if mask is None:
            return
        mask = mask.repeat(1, self.h).view(query.shape[0], -1)
        mask = mask.unsqueeze(dim=-1)
        return mask  # of shape [B * h, Mq, 1]

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
            mask: Union[Tensor, None] = None,
            query_mask: Union[Tensor, None] = None,
            key_mask: Union[Tensor, None] = None
            ) -> Tuple[Tensor, Tensor]:
        """Performs multi-head attention on the provided key, query and value
        Args:
            key (Tensor): The key tensor of shape [B, Mt, d_model]
            query (Tensor): The query tensor of shape [B, Ms, d_model]
            value (Tensor): The value tensor of shape [B, Mt, d_model]
            mask (Union[Tensor, None]): The input mask of shape [B, Ms]
        Returns:
            Tuple[Tensor, Tensor]: A tuple of the attention matrix and the
            results after performing multi-head attention where the first of
            shape [h, B, Ms, Mt] and the second of shape [B, Tq, d_model].
        """
        [b, s, _] = query.shape
        K = self.fc_key(key)
        Q = self.fc_query(query)
        V = self.fc_value(value)
        (Q, K, V) = self._change_dim(Q, K, V)  # [h * B, T, dk]
        K = K.permute(0, 2, 1)  # [h, T, B, dk]
        att, result = self.perform_att(
            Q, K, V, mask=mask, query_mask=query_mask, key_mask=key_mask
            )
        result = result.view(b, self.h, s, self.dk)
        result = result.permute(0, 2, 1, 3)
        result = result.contiguous().view(b, s, -1)
        result = torch.cat([query, result], dim=-1)
        result = self.proj_fc(result)
        out = self.dropout(result)
        return att, out


class MultiHeadSelfAtt(MultiHeadAtt):
    """Implements the multi-head self attention module

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        p_dropout (float): The dropout ratio.
        device (str): The device to map the operations to.
    """
    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__(d_model, h, p_dropout, device)

    def get_mask(
            self,
            query: Tensor,
            key: Tensor,
            mask: Union[None, Tensor],
            *args, **kwargs
            ):
        # Query of shape [B * h, Mq, d_model]
        # mask of shape [B, Mq] or None
        if mask is None:
            return
        max_len = mask.shape[1]
        # [B * h, M, M]
        mask = super().get_mask(query, key, mask).squeeze()
        mask = mask.repeat(1, max_len).view(query.shape[0], max_len, max_len)
        # don't look ahead mask of shape [B*h, Mq, Mk]
        la_mask = self.get_square_mask(query, query)
        mask = la_mask.to(self.device) | mask.to(self.device)
        mask = torch.cumsum(mask, dim=-1) >= 2
        return mask

    def get_square_mask(self, query: Tensor, key: Tensor) -> Tensor:
        mask = torch.triu(torch.ones(query.shape[1], key.shape[1]))
        mask = mask.type(torch.BoolTensor)
        mask = mask.unsqueeze(0)
        mask = mask.repeat(query.shape[0], 1, 1)
        return mask


class FeedForward(nn.Module):
    """Implements the feedforward Module in the model, where the input is
    scaled to a hidden_size and then back to the d_model.

    Args:
        d_model (int): The model dimensionality.
        hidden_size (int): the hidden size of the module.
        p_dropout (float): The dropout ratio.
    """
    def __init__(
            self,
            d_model: int,
            hidden_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=d_model,
            out_features=hidden_size
        )
        self.fc2 = nn.Linear(
            in_features=hidden_size,
            out_features=d_model
        )
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class AddAndNorm(nn.Module):
    """Implements the Add & Norm module where the input of the last module
    and the output of the last module added and then fed to Layernorm

    Args:
        d_model (int): The model dimensionality.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lnrom = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, out: Tensor):
        return self.lnrom(x + out)


class EncoderLayer(nn.Module):
    """Implements the basic unit of the encoder and it contains the below:
        - multi-head self attention layer.
        - feed forward layer.
        - Residual add and layer normalization after each of the above.

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        hidden_size (int): the hidden size of the feed forward module.
        p_dropout (float): The dropout ratio.
        device (str): the device to map the operations to.
    """
    def __init__(
            self,
            d_model: int,
            h: int,
            hidden_size: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        self.mhsa = MultiHeadAtt(
            d_model=d_model,
            h=h,
            p_dropout=p_dropout,
            device=device
            )
        self.mhsa_add_and_norm = AddAndNorm(
            d_model=d_model
            )
        self.ff = FeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            p_dropout=p_dropout
        )
        self.ff_add_and_norm = AddAndNorm(
            d_model=d_model
        )

    def forward(self, x: Tensor, mask: Union[Tensor, None]) -> Tensor:
        """Given the input of shape [B, M, d] performs self attention
        on the input and return back the result of shape [B, M, d]

        Args:
            x (Tensor): The input of shape [B, M, d]
            mask Union[Tensor, None]: The input mask of shape [B, M]

        Returns:
            Tensor: The result out of the self attention of shape [B, M, d]
        """
        _, out = self.mhsa(x, x, x, query_mask=mask, key_mask=mask)
        out = self.mhsa_add_and_norm(x, out)
        ff_out = self.ff(out)
        out = self.ff_add_and_norm(out, ff_out)
        return out


class DecoderLayer(nn.Module):
    """Implements the basic unit of the decoder

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        p_dropout (float): The dropout ratio.
        hidden_size (int): the hidden size of the feed forward module.
        device (str): the device to map the operations to.
    """
    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            hidden_size: int,
            device: str
            ) -> None:
        super().__init__()
        self.mhsa = MultiHeadSelfAtt(
            d_model=d_model,
            h=h,
            p_dropout=p_dropout,
            device=device
        )
        self.add_and_norm_1 = AddAndNorm(d_model=d_model)
        self.mha = MultiHeadAtt(
            d_model=d_model,
            h=h,
            p_dropout=p_dropout,
            device=device
        )
        self.add_and_norm_2 = AddAndNorm(d_model=d_model)
        self.ff = FeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            p_dropout=p_dropout
        )
        self.add_and_norm_3 = AddAndNorm(d_model=d_model)

    def forward(
            self,
            x: Tensor,
            encoder_values: Tensor,
            mask: Union[Tensor, None] = None,
            query_mask: Union[Tensor, None] = None,
            key_mask: Union[Tensor, None] = None
            ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pass the data into the decoder blocks which they are:
        - MMHA
        - ADD & NORM
        - MHA
        - ADD & NORM
        - Feed Forward
        - ADD & NORM

        Args:
            x (Tensor): The input tensor of shape [B, Td, d_model]
            encoder_values (Tensor): The encoder results of shape
            [B, Me, d_model]
            mask (Union[Tensor, None]): The input mask of shape [B, M].
            Default None.

        Returns:
            Tuple[Tensor, Tensor]: a tuple of the results,
            the output and attention weights.
        """
        _, out = self.mhsa(x, x, x, mask=mask)
        out_1 = self.add_and_norm_1(x, out)
        att, out = self.mha(
            query=out_1,
            key=encoder_values,
            value=encoder_values,
            query_mask=query_mask,
            key_mask=key_mask
            )
        out = self.add_and_norm_2(out_1, out)
        out_1 = self.ff(out)
        out = self.add_and_norm_3(out_1, out)
        return out, att


class PositionalEmb(nn.Module):
    """Implements the positional Embedding Module
    Args:
        voc_size (int): The number of covered vocabulary.
        d_model (int): The model dimensionality.
        pad_idx (int): The padding index to zero out its embedding.
        device (str): The device to map the operations to.
    """
    def __init__(
            self,
            voc_size: int,
            d_model: int,
            pad_idx: int,
            device: str
            ) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )
        self.device = device
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        max_len = x.shape[-1]
        out = self.emb(x)
        pe = get_positionals(max_len, self.d_model).to(self.device)
        return out + pe


class EncoderLayers(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layers: int,
            voc_size: int,
            hidden_size: int,
            h: int,
            p_dropout: float,
            pad_idx: int,
            device: str
            ) -> None:
        super().__init__()
        self.emb = PositionalEmb(
            voc_size=voc_size,
            d_model=d_model,
            pad_idx=pad_idx,
            device=device
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                h=h,
                hidden_size=hidden_size,
                p_dropout=p_dropout,
                device=device
            )
            for _ in range(n_layers)
        ])

    def forward(
            self, x: Tensor, mask: Union[Tensor, None]
            ) -> Tensor:
        out = self.emb(x)
        for layer in self.layers:
            out = layer(out, mask=mask)
        return out


class DecoderLayers(nn.Module):
    def __init__(
            self,
            voc_size: int,
            d_model: int,
            n_layers: int,
            h: int,
            p_dropout: float,
            hidden_size: int,
            pad_idx: int,
            device: str
            ) -> None:
        super().__init__()
        self.emb = PositionalEmb(
            voc_size=voc_size,
            d_model=d_model,
            pad_idx=pad_idx,
            device=device
        )
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                h=h,
                p_dropout=p_dropout,
                hidden_size=hidden_size,
                device=device
            )
            for _ in range(n_layers)
        ])

    def forward(
            self,
            x: Tensor,
            mask: Tensor,
            enc_values: Tensor,
            key_mask: Union[Tensor, None] = None
            ):
        out = self.emb(x)
        for layer in self.layers:
            out, att = layer(
                x=out,
                encoder_values=enc_values,
                mask=mask,
                query_mask=mask,
                key_mask=key_mask
                )
        return out, att


class PackedGRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bidirectional: bool,
            padding_value: Union[float, int],
            num_layers=1
            ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.padding_value = padding_value
        self.num_layers = num_layers

    def forward(self, x: Tensor, lengths: List[int], hn=None) -> Tensor:
        packed_seq = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
            )
        if hn is None:
            hn = torch.zeros(
                self.num_layers,
                x.shape[0],
                self.hidden_size
                ).to(x.device)
        output, hn = self.gru(packed_seq, hn)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        return output, hn


class GRUBlock(nn.Module):
    def __init__(
            self,
            inp_size: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_value: Union[float, int]
            ) -> None:
        super().__init__()
        self.gru = PackedGRU(
            input_size=inp_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            padding_value=padding_value
        )

        self.ff = FeedForward(
            d_model=hidden_size if bidirectional is False else 2 * hidden_size,
            hidden_size=2 * hidden_size if bidirectional is False else 4 * hidden_size,
            p_dropout=p_dropout
        )
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p_dropout)
        self.lnorm = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(
            self, x: Tensor, lengths: List[int], hn=None
            ) -> Tuple[Tensor, Tensor]:
        out, h = self.gru(x, lengths, hn=hn)
        out = self.dropout(out)
        out = self.ff(out)
        out = self.lnorm(out)
        return out, h


class GRUStack(nn.Module):
    def __init__(
            self,
            n_layers: int,
            inp_size: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_value: Union[float, int]
            ) -> None:
        super().__init__()
        self.grus = nn.ModuleList([
            GRUBlock(
                inp_size=inp_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                p_dropout=p_dropout,
                bidirectional=bidirectional,
                padding_value=padding_value
            )
            for i in range(n_layers)
        ])
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, lengths: List[int], hn=None) -> Tensor:
        out = x
        hns = []
        for i, layer in enumerate(self.grus):
            if hn is not None:
                out, h = layer(
                    out,
                    lengths,
                    hn=hn if hn.shape[0] != len(self.grus) else hn[i:i+1, ...]
                    )
            else:
                out, h = layer(out, lengths, hn=hn)
            hns.append(h)
        hns = torch.vstack(hns)
        return out, hns


class RNNEncoder(nn.Module):
    def __init__(
            self,
            voc_size: int,
            emb_size: int,
            n_layers: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_idx: int,
            padding_value: Union[float, int],
            ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=emb_size,
            padding_idx=padding_idx
        )
        self.gru_stack = GRUStack(
            n_layers=n_layers,
            inp_size=emb_size,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=bidirectional,
            padding_value=padding_value
        )

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        out = self.embedding(x)
        out, hn = self.gru_stack(out, lengths)
        return out, hn


class Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=2 * hidden_size,
            out_features=hidden_size
        )

    def forward(self, query, key, value):
        query = query.permute(1, 0, 2)
        key = key.permute(0, 2, 1)
        e = torch.softmax(torch.matmul(query, key), dim=-1)
        result = torch.matmul(e, value)
        if result.shape[0] != query.shape[0]:
            query = query.repeat(result.shape[0], 1, 1)
        result = torch.cat([result, query], dim=-1)
        result = self.fc(result)
        result = result.permute(1, 0, 2)
        return result, e


class RNNDecoder(nn.Module):
    def __init__(
            self,
            max_len: int,
            voc_size: int,
            emb_size: int,
            n_layers: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_idx: int,
            padding_value: Union[float, int]
            ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=emb_size,
            padding_idx=padding_idx
        )
        self.gru_stack = GRUStack(
            n_layers=n_layers,
            inp_size=emb_size,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=False,
            padding_value=padding_value
        )
        self.pred_fc = nn.Linear(
            in_features=hidden_size,
            out_features=voc_size
        )
        self.max_len = max_len
        self.attention = Attention(
            hidden_size=hidden_size
            )
        self.key_fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )
        self.value_fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )
        self.query_fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )

    def _process_query(self, h: Tensor):
        h = h.permute(1, 0, 2)
        h = h.contiguous().view(h.shape[0], 1, -1)
        h = self.query_fc(h)
        h = h.permute(1, 0, 2)
        return h

    def forward(
            self,
            enc_values: Tensor,
            hn: Tensor,
            x: Tensor,
            lengths: Tensor
            ) -> Tensor:
        max_len = lengths.max().item()
        out = self.embedding(x)
        key = self.key_fc(enc_values)
        value = self.value_fc(enc_values)
        attention = []
        result = []
        for i in range(max_len):
            step_lens = torch.ones(x.shape[0], dtype=torch.long)
            hn = self.query_fc(hn)
            hn, att = self.attention(key=key, value=value, query=hn)
            output, hn = self.gru_stack(
                out[..., i:i+1, :], lengths=step_lens, hn=hn
                )
            result.append(output)
            # We can return all layers' attention rather than the last one!
            attention.append(att[:, -1:, :])
        result = torch.hstack(result)
        attention = torch.hstack(attention)
        result = self.pred_fc(result)
        return result, attention

    def predict(self, hn, x, enc_values, key=None, value=None):
        out = self.embedding(x)
        step_lens = torch.ones(x.shape[0], dtype=torch.long)
        if enc_values is not None:
            key = self.key_fc(enc_values)
            value = self.value_fc(enc_values)
        hn = self.query_fc(hn)
        hn, att = self.attention(key=key, value=value, query=hn)
        output, hn = self.gru_stack(out, lengths=step_lens, hn=hn)
        result = self.pred_fc(output)
        return hn, att, result, key, value
