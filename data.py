from functools import partial
from typing import Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator


def _yield_tokens(data_iter: Iterable, language: str, src_lang: str, tgt_lang: str, token_trans: dict) -> List[str]:
    language_index = {src_lang: 0, tgt_lang: 1}

    for data_sample in data_iter:
        yield token_trans[language](data_sample[language_index[language]])


def _sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def _tensor_transform(token_ids: List[int], bos_idx: int, eos_idx: int):
    return torch.cat(
        (
            torch.tensor([bos_idx]),
            torch.tensor(token_ids),
            torch.tensor([eos_idx])
        )
    )


def create_masks(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(-2)

    tgt_mask = None
    if tgt is not None:
        pad_tgt_mask = (tgt != pad_idx).unsqueeze(-2)
        time_mask = torch.triu(torch.ones((pad_tgt_mask.shape[-1], pad_tgt_mask.shape[-1])), diagonal=1) == 0
        time_mask = time_mask.type_as(pad_tgt_mask.data)
        tgt_mask = pad_tgt_mask & time_mask

    return src_mask, tgt_mask


class DeEnSetGenerator:
    def __init__(self):
        multi30k.URL["train"] = \
            "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
        multi30k.URL["valid"] = \
            "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

        self.src_language = "de"
        self.tgt_language = "en"
        self.special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.unk_idx, self.pad_idx, self.bos_idx, self.eos_idx = 0, 1, 2, 3

        self.token_transform = {
            self.src_language: get_tokenizer("spacy", language="de_core_news_sm"),
            self.tgt_language: get_tokenizer("spacy", language="en_core_web_sm")
        }
        self._create_vocab_transform()
        self._create_text_transform()
        self.partial_collate = partial(
            self.collate_fn,
            txt_transforms=self.text_transforms,
            src_lang=self.src_language,
            tgt_lang=self.tgt_language,
            pad_idx=self.pad_idx
        )

    def _create_vocab_transform(self):
        self.vocab_transform = {}
        for ln in [self.src_language, self.tgt_language]:
            train_iter = Multi30k(split="train", language_pair=(self.src_language, self.tgt_language))
            self.vocab_transform[ln] = build_vocab_from_iterator(
                _yield_tokens(train_iter, ln, self.src_language, self.tgt_language, self.token_transform),
                min_freq=1,
                specials=self.special_symbols,
                special_first=True
            )

        for ln in [self.src_language, self.tgt_language]:
            self.vocab_transform[ln].set_default_index(self.unk_idx)

    def _create_text_transform(self):
        self.text_transforms = {}
        partial_transform = partial(
            _tensor_transform,
            bos_idx=self.bos_idx,
            eos_idx=self.eos_idx
        )
        for ln in [self.src_language, self.tgt_language]:
            self.text_transforms[ln] = _sequential_transforms(
                self.token_transform[ln],
                self.vocab_transform[ln],
                partial_transform
            )

    @staticmethod
    def collate_fn(batch, txt_transforms, src_lang, tgt_lang, pad_idx):
        src_batch, tgt_batch = [], []
        src_seq_len, tgt_seq_len = [], []
        for src_sample, tgt_sample in batch:
            src_data = txt_transforms[src_lang](src_sample.rstrip("\n"))
            tgt_data = txt_transforms[tgt_lang](tgt_sample.rstrip("\n"))
            src_batch.append(
                src_data
            )
            src_seq_len.append(len(src_data))
            tgt_batch.append(
                tgt_data
            )
            tgt_seq_len.append(len(tgt_data))
        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch, src_seq_len, tgt_seq_len

    def get_set(self, batch_size, mode="train", shuffle=True):
        assert mode in ["train", "valid"]
        _iter = Multi30k(split=mode, language_pair=(self.src_language, self.tgt_language))
        return DataLoader(
            _iter,
            batch_size=batch_size,
            collate_fn=self.partial_collate,
            shuffle=shuffle
        )
