import torch
from torch.nn.utils.rnn import pad_sequence


class DeEnTranslator:
    def __init__(self, model, set_generator, temperature):
        self.set_gen = set_generator
        self.model = model
        self.temperature = temperature

        self._src_lang = "de"
        self._tgt_lang = "en"

    def generate(self, sentences, max_out_len=None):
        self.model.eval()

        max_out_len = max_out_len if max_out_len is not None else max([len(x.split(" ")) for x in sentences]) + 7

        de_tokens = [self.set_gen.text_transforms[self._src_lang](x) for x in sentences]

        de_tokens_lens = [x.shape[0] for x in de_tokens]
        de_tokens = torch.transpose(pad_sequence(de_tokens, padding_value=self.set_gen.pad_idx), 1, 0)

        with torch.no_grad():
            en_tokens, _ = self.model.forward_gen(
                de_tokens, de_tokens_lens, max_out_len, self.set_gen.bos_idx, self.temperature
            )

        return self.parse_tokens(en_tokens, self.set_gen.vocab_transform[self._tgt_lang])


def parse_tokens(tokens, vocab_transform, bos_symbol="<bos>", eos_symbol="<eos>"):
    out = []
    for sent_idx in range(tokens.shape[0]):
        sentence = vocab_transform.lookup_tokens(
            tokens[sent_idx, :].cpu().numpy().tolist()
        )[1:]
        try:
            bos_idx = sentence.index(bos_symbol)
            sentence = sentence[bos_idx + 1:]
        except ValueError:
            pass
        try:
            eos_idx = sentence.index(eos_symbol)
            sentence = sentence[:eos_idx]
        except ValueError:
            pass
        out.append(" ".join(sentence))
    return out


def _load_model_and_set(path_to_model):
    import config
    from data import DeEnSetGenerator
    from transformer.blocks import TransformerModel

    loader = DeEnSetGenerator()
    model = TransformerModel(
        **{
            "dec_embed_size": len(loader.vocab_transform["en"]),
            "enc_embed_size": len(loader.vocab_transform["de"]),
            "padding_idx": loader.pad_idx,
            **config.model_hyperparams
        }
    )
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, loader


def main():
    path_to_model = ""
    temperature = 1.
    model, dataloader = _load_model_and_set(path_to_model)
    translator = DeEnTranslator(
        model,
        dataloader,
        temperature
    )

    print("Type German sentence and prepare for a translation!!!")
    while True:
        german_sentence = input("Type the sentence: ")
        print(
            translator.generate([german_sentence])[0]
        )


if __name__ == '__main__':
    main()
