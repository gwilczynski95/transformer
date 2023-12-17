from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from nltk.translate.bleu_score import corpus_bleu
# from nltk.translate.meteor_score import meteor_score

import config


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
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=optimizer_params["lr"])

        start_epoch = 0
        if checkpoint_path:
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
            references = []
            hypotheses = []
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                output_texts = self.parse_output_text(outputs)
                label_texts = self.parse_output_text(inputs)

                references.append(label_texts)
                hypotheses.append(output_texts)

            avg_loss = running_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

            # Calculate and log training metrics
            train_bleu, train_meteor = self._calculate_metrics(references, hypotheses)
            self.log_metrics(epoch, "Train", avg_loss, train_bleu, train_meteor)

            # Save checkpoint
            save_path = Path(self.model_dir, "checkpoints", f"step_{epoch + 1}")
            self.save_checkpoint(epoch, save_path, optimizer)

            # Evaluate on validation set if available
            if self.val_loader is not None:
                val_bleu, val_meteor = self.calculate_metrics(self.val_loader)
                self.log_metrics(epoch, "Validation", avg_loss, val_bleu, val_meteor)

    def calculate_metrics(self, data_loader):
        self.model.eval()
        references = []
        hypotheses = []
        with torch.no_grad():
            for src, tgt, src_lens, tgt_lens in data_loader:
                enc_x = self.model.encoder(src, src_lens)
                output_tokens = torch.full([src.shape[0], 1], self.set_loader.bos_idx, dtype=torch.int64)
                out_token_lens = [1] * src.shape[0]
                for _ in range(tgt.shape[1]):
                    dec_out = self.model.decoder(output_tokens, out_token_lens, enc_x, None)
                    _p = torch.softmax(dec_out / self.temperature, dim=-1).detach().numpy()[0, -1, :]
                    token = np.random.choice(np.arange(len(_p)), p=_p)
                    output_tokens = torch.cat([output_tokens, token], dim=-1)  # todo: fix this 
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                # todo: calculate the loss after the last outputs
                # Convert outputs and labels to text
                output_texts = self.parse_output_text(outputs)
                label_texts = self.parse_output_text(inputs)

                references.append(label_texts)
                hypotheses.append(output_texts)

        bleu_score, meteor_average = self._calculate_metrics(references, hypotheses)

        return bleu_score, meteor_average

    def save_checkpoint(self, epoch, path, optimizer):
        if path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{path}_epoch_{epoch}.pt')

    def parse_output_text(self, sequence):
        return None

    # @staticmethod
    # def _calculate_metrics(references, hypotheses):
    #     bleu_score = corpus_bleu(references, hypotheses)
    #     meteor_scores = [meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    #     meteor_average = sum(meteor_scores) / len(meteor_scores)
    #     return bleu_score, meteor_average

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
