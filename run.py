import config
from data import DeEnSetGenerator
from trainer import Trainer
from transformer.blocks import TransformerModel


def main():
    loader = DeEnSetGenerator()

    model = TransformerModel(
        **{
            "dec_embed_size": len(loader.vocab_transform["en"]),
            "enc_embed_size": len(loader.vocab_transform["de"]),
            "padding_idx": loader.pad_idx,
            **config.model_hyperparams
        }
    )

    train_loader = loader.get_set(
        config.training_hyperparams["batch_size"],
        mode="train",
        shuffle=True
    )
    val_loader = loader.get_set(
        config.training_hyperparams["batch_size"],
        mode="valid",
        shuffle=False
    )

    # Create the trainer and start training
    trainer = Trainer(model, train_loader, val_loader, device='cuda')
    trainer.train(
        config.training_hyperparams["epochs"],
        config.training_hyperparams["optimizer"],
        config.checkpoint_path
    )


if __name__ == "__main__":
    main()
