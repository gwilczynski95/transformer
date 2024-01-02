from pathlib import Path
import shutil

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

    # Create the trainer and start training
    model_dir = Path(
        config.experiment_parent_path,
        config.experiment_name
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(config.__file__),
        Path(model_dir, "config.py")
    )
    trainer = Trainer(
        model_dir,
        model,
        loader,
        device='cuda'
    )
    trainer.train(
        config.training_hyperparams["epochs"],
        config.training_hyperparams["checkpoint_path"],
    )


if __name__ == "__main__":
    main()
