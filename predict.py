from typing import List
import torch
from torch.nn import Module
from torch import Tensor, BoolTensor
from interfaces import IPredictor, IProcessor, ITokenizer


class BasePredictor(IPredictor):

    def __init__(
            self,
            model: Module,
            tokenizer: ITokenizer,
            max_len: int,
            processor: IProcessor,
            device: str
            ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sos = tokenizer.special_tokens.sos_id
        self.eos = tokenizer.special_tokens.eos_id
        self.processor = processor
        self.device = device

    def is_terminated(self, preds: Tensor) -> BoolTensor:
        """Checks whether the sentence reached an end
        or not by checking if if reached the maximum length
        or EOS token emitted.

        Args:
            preds (Tensor): The prediction tensor of shape [B, M]

        Returns:
            BoolTensor: BoolTensor shows which sentences terminated.
        """
        preds_lens = preds.shape[-1]
        if preds_lens >= self.max_len:
            return torch.ones(preds_lens, dtype=torch.bool)
        return preds[..., -1] == self.eos

    def process_text(self, sentence: str) -> Tensor:
        sentence = self.processor.run([sentence])[0]
        assert len(sentence) != 0, 'The cleaned sentence\'s length is Zeros!'
        tokens = self.tokenizer.tokenize(sentence, add_sos=True, add_eos=True)
        input = torch.LongTensor([tokens])
        input = input.to(self.device)
        return input

    def get_dec_start(self, length=1) -> Tensor:
        input = torch.LongTensor([[self.sos]])
        input = input.repeat(length, 1)
        input = input.to(self.device)
        return input

    def finalize(self, results: Tensor):
        results = results[0].tolist()
        results = results[1:]
        if self.eos in results:
            results = results[:results.index(self.eos)]
        results = self.tokenizer.ids2tokens(results)
        return ''.join(results)


class GreedyPredictor(BasePredictor):
    def __init__(
            self,
            model: Module,
            tokenizer: ITokenizer,
            max_len: int,
            processor: IProcessor,
            device: str
            ) -> None:
        super().__init__(model, tokenizer, max_len, processor, device)

    @torch.no_grad()
    def predict(self, sentence: str):
        enc_inp = self.process_text(sentence)
        dec_inp = self.get_dec_start()
        while self.is_terminated(dec_inp)[0].item() is False:
            enc_inp, preds, _ = self.model.predict(
                enc_inp=enc_inp,
                dec_inp=dec_inp,
                enc_mask=None,
                dec_mask=torch.zeros(1, 1, dtype=torch.bool)
                )
            preds = torch.argmax(preds, dim=-1)
            preds = preds[0, -1]
            preds = preds.view(1, 1)
            dec_inp = torch.cat([dec_inp, preds], dim=-1)
        return self.finalize(dec_inp)


class BeamPredictor(BasePredictor):
    def __init__(
            self,
            model: Module,
            tokenizer: ITokenizer,
            max_len: int,
            processor: IProcessor,
            device: str,
            beam_width: int,
            alpha: int
            ) -> None:
        super().__init__(
            model, tokenizer, max_len, processor, device
            )
        self.alpha = alpha
        self.beam_width = beam_width

    def _get_updated_item(
            self, item: tuple, idx: Tensor, log_p: Tensor
            ) -> tuple:
        preds = torch.cat([item[0], idx.view(1, 1)], dim=-1)
        length = preds.shape[-1]
        score = item[1] + log_p.item()
        norm_score = score / (length ** self.alpha)
        return preds, score, norm_score

    def predict(self, sentence: str):
        enc_inp = self.process_text(sentence)
        # [Prediction, score, normalized score]
        in_progress = [(self.get_dec_start(), 0, 0)]
        completed = []
        counter = 0
        while len(in_progress) != 0 and counter <= self.max_len:
            temp = []
            for item in in_progress:
                enc_inp, preds, _ = self.model.predict(
                    enc_inp=enc_inp,
                    dec_inp=item[0],
                    enc_mask=None,
                    dec_mask=torch.zeros(1, 1, dtype=torch.bool)
                    )
                values, indices = torch.topk(preds, k=self.beam_width, dim=-1)
                for log_p, idx in zip(values, indices):
                    temp.append(self._get__updated_item(item, idx, log_p))
                temp = sorted(temp, key=lambda x: x[-1], reverse=True)
                temp = temp[:self.beam_width - len(completed)]
                completed.extend(
                    list(
                        filter(lambda x: x[0][0][-1].item() == self.eos, temp)
                        )
                    )
                in_progress = list(
                    filter(lambda x: x[0][0][-1].item() != self.eos, temp)
                    )
        results = max(completed, key=lambda x: x[-1])
        return self.finalize(results[0])


class BatchPredictor(BasePredictor):
    def __init__(
            self,
            model: Module,
            tokenizer: ITokenizer,
            max_len: int,
            processor: IProcessor,
            device: str,
            *args, **kwargs
            ) -> None:
        super().__init__(model, tokenizer, max_len, processor, device)
        self.pad = tokenizer.special_tokens.pad_id

    def pad_items(self, items: List[List[int]]):
        results = []
        masks = []
        max_len = max([len(item) for item in items])
        for item in items:
            diff = max_len - len(item)
            mask = [False] * len(item) + [True] * diff
            item = item + [self.pad] * diff
            results.append(item)
            masks.append(mask)
        return results, masks

    def process_text(self, sentences: List[str]) -> Tensor:
        sentences = self.processor.run(sentences)
        items = self.tokenizer.batch_tokenizer(
            sentences, add_sos=True, add_eos=True
            )
        items, mask = self.pad_items(items)
        input = torch.LongTensor(items)
        mask = torch.BoolTensor(mask)
        input = input.to(self.device)
        mask = mask.to(self.device)
        return input, mask

    @torch.no_grad()
    def predict(self, sentences: List[str]):
        enc_inp, mask = self.process_text(sentences)
        dec_inp = self.get_dec_start(length=len(sentences))
        is_terminated = torch.zeros(len(sentences), dtype=torch.bool)
        while is_terminated.sum().item() != len(sentences):
            enc_inp, preds, _ = self.model.predict(
                enc_inp=enc_inp,
                dec_inp=dec_inp,
                enc_mask=mask,
                dec_mask=torch.zeros_like(
                    dec_inp, dtype=torch.bool
                    ).to(self.device)
                )
            preds = torch.argmax(preds, dim=-1)
            preds = preds[:, -1]
            preds = preds.view(-1, 1)
            dec_inp = torch.cat([dec_inp, preds], dim=-1)
            is_terminated = is_terminated | self.is_terminated(preds).cpu()
        return list(map(self.finalize, torch.unsqueeze(dec_inp, dim=1)))
