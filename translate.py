import numpy as np
import torch


class DeEnTranslator:
    def __init__(self, model, set_generator, temperature):
        self.set_gen = set_generator
        self.model = model
        self.temperature = temperature

        self._src_lang = "de"
        self._tgt_lang = "en"

    def generate(self, sentence):
        max_out_len = len(sentence) + 5

        de_tokens = self.set_gen.text_transforms[self._src_lang](sentence)[np.newaxis]
        en_tokens = torch.tensor([[self.set_gen.bos_idx]], dtype=torch.int64)

        de_tokens_lens = [de_tokens.shape[-1]]
        en_tokens_lens = [1]

        enc_x = self.model.encoder(de_tokens, de_tokens_lens)

        for _ in range(max_out_len):
            dec_out = self.model.decoder(en_tokens, en_tokens_lens, enc_x, None)
            _p = torch.softmax(dec_out / self.temperature, dim=-1).detach().numpy()[0, -1, :]
            token = np.random.choice(np.arange(len(_p)), p=_p)
            en_tokens = torch.cat([en_tokens, torch.tensor([[token]])], dim=-1)
            en_tokens_lens[0] += 1
            if token == self.set_gen.eos_idx:
                break

        return " ".join(
            self.set_gen.vocab_transform[self._tgt_lang].lookup_tokens(en_tokens.cpu().numpy()[0].tolist())
        ).replace("<bos>", "").replace("<eos>", "")
