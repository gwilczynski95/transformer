import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, device=None, log_dir='runs/trainer'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir)

    def train(self, epochs, optimizer_params, checkpoint_path=None):
        # todo: add proper optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        start_epoch = 0
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

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
            self.log_metrics(epoch, avg_loss, train_bleu, train_meteor, 'Train')

            # Save checkpoint
            self.save_checkpoint(epoch, checkpoint_path, optimizer)

            # Evaluate on validation set if available
            if self.val_loader is not None:
                val_bleu, val_meteor = self.calculate_metrics(self.val_loader)
                self.log_metrics(epoch, avg_loss, val_bleu, val_meteor, 'Validation')

    def calculate_metrics(self, data_loader):
        self.model.eval()
        references = []
        hypotheses = []
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

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

    @staticmethod
    def _calculate_metrics(references, hypotheses):
        bleu_score = corpus_bleu(references, hypotheses)
        meteor_scores = [meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        meteor_average = sum(meteor_scores) / len(meteor_scores)
        return bleu_score, meteor_average

    def log_metrics(self, epoch, loss, bleu, meteor, phase):
        self.writer.add_scalar(f'{phase}/Loss', loss, epoch)
        self.writer.add_scalar(f'{phase}/BLEU', bleu, epoch)
        self.writer.add_scalar(f'{phase}/METEOR', meteor, epoch)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
