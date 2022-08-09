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
        sentence = self.processor.run(sentence)
        assert len(sentence) != 0, 'The cleaned sentence\'s length is Zeros!'
        tokens = self.tokenizer.tokenize(sentence, add_eos=True)
        input = torch.LongTensor([tokens])
        input = input.to(self.device)
        return input

    def get_dec_start(self) -> Tensor:
        input = torch.LongTensor([[self.sos]])
        input = input.to(self.device)
        return input

    def finalize(self, results: Tensor):
        results = results[0].tolist()
        results = results[1:]
        if results[-1] == self.eos:
            results = results[:-1]
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
            preds, _ = self.model(
                enc_inp=enc_inp,
                dec_inp=dec_inp,
                enc_mask=None,
                dec_mask=None
                )
            preds = torch.argmax(preds, dim=-1)
            preds = preds[0, -1]
            preds = preds.view(1, 1)
            dec_inp = torch.cat([dec_inp, preds], dim=-1)
        return self.finalize(dec_inp)
