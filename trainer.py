from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional.text import bleu_score
from nltk import word_tokenize
from nltk.translate.meteor_score import meteor_score

import config
from translate import parse_tokens


class Trainer:
    def __init__(self, model_dir, model, set_loader, device=None):
        self.model_dir = model_dir
        self.model = model
        self.set_loader = set_loader
        self.train_loader = set_loader.get_set(
            config.batch_size,
            mode="train",
            shuffle=True
        )
        self.val_loader = set_loader.get_set(
            config.batch_size,
            mode="valid",
            shuffle=False
        )
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.writer = SummaryWriter(
            Path(model_dir, "logs")
        )

    def train(self, epochs, optimizer_params, checkpoint_path=None):
        # todo: add proper optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.set_loader.pad_idx)
        optimizer = optim.Adam(self.model.parameters(), lr=optimizer_params["lr"])
        print("Start training")

        start_epoch = 0
        if checkpoint_path:
            print("Load checkpoint")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

            self.model_dir = Path(checkpoint_path.parent.parent)
            self.writer = SummaryWriter(
                Path(self.model_dir, "logs")
            )

        for epoch in range(start_epoch, epochs):
            self.model.train()
            running_loss = 0.0
            _iters = 0
            for src, tgt, src_lens, tgt_lens in self.train_loader:
                src = src.to(self.device).T
                tgt = tgt.to(self.device).T
                tgt_input = tgt[:, :-1]
                tgt = tgt[:, 1:]

                optimizer.zero_grad()
                outputs = self.model(src, tgt_input, src_lens, tgt_lens)
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt.reshape(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _iters += 1

            avg_loss = running_loss / _iters
            print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

            self.log_metrics(epoch, "Train", avg_loss)

            # Save checkpoint
            save_path = Path(self.model_dir, "checkpoints", f"step_{epoch + 1}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(epoch, save_path, optimizer)

            # Evaluate on validation set if available
            if self.val_loader is not None:
                val_loss, val_bleu, val_meteor = self.calculate_metrics(self.val_loader, criterion)
                self.log_metrics(epoch, "Validation", val_loss, val_bleu, val_meteor)

    def calculate_metrics(self, data_loader, loss_fn, temperature=1.):
        self.model.eval()
        references = []
        hypotheses = []
        running_loss = 0.0
        with torch.no_grad():
            _iters = 0
            for src, tgt, src_lens, tgt_lens in data_loader:
                src = src.to(self.device).T
                tgt = tgt.to(self.device).T
                tgt = tgt[:, 1:]
                out_tokens, out_probas = self.model.forward_gen(
                    src, src_lens, max(tgt_lens) - 1, self.set_loader.bos_idx, temperature
                )
                output_texts = parse_tokens(out_tokens, self.set_loader.vocab_transform["en"])
                label_texts = parse_tokens(tgt, self.set_loader.vocab_transform["en"])

                references.extend(label_texts)
                hypotheses.extend(output_texts)

                loss = loss_fn(out_probas.reshape(-1, out_probas.shape[-1]), tgt.reshape(-1))
                running_loss += loss.item()
                _iters += 1

        running_loss /= _iters
        bleu_res, meteor_average = self._calculate_metrics(references, hypotheses)

        return running_loss, bleu_res, meteor_average

    def save_checkpoint(self, epoch, path, optimizer):
        if path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{path}_epoch_{epoch}.pt')

    @staticmethod
    def _calculate_metrics(references, hypotheses):
        bleu_res = bleu_score(hypotheses, [[x] for x in references]).item()
        meteor_scores = [
            meteor_score([word_tokenize(ref)], word_tokenize(hyp)) for hyp, ref in zip(hypotheses, references)
        ]
        meteor_average = sum(meteor_scores) / len(meteor_scores)
        return bleu_res, meteor_average

    def log_metrics(self, epoch, phase, loss, bleu=None, meteor=None):
        self.writer.add_scalar(f'{phase}/Loss', loss, epoch)
        if bleu is not None:
            self.writer.add_scalar(f'{phase}/BLEU', bleu, epoch)
        if meteor is not None:
            self.writer.add_scalar(f'{phase}/METEOR', meteor, epoch)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
