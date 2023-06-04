from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from trainer import GithubCollabTrainer
import torch
from argparse import Namespace

hparams = {
    "input_size": 33,
    "hidden_size": 128,
    "num_layers": 2,
    "learning_rate": 0.0005,
    "batch_size": 128,
    "num_epochs": 10,
    "seq_len": 8,
    "weight_decay": 0.01,
}

# convert haprams to namespace
hparams = Namespace(**hparams)

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(hparams):
    print(hparams.input_size)
    print("hi")
    model = GithubCollabTrainer(hparams)
    # add model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_epoch",
        mode="max",
        save_top_k=1,
        dirpath="../models/checkpoints",
        filename="github_collab_lstm-{epoch:02d}-{val_f1_epoch:.2f}",
    )
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)


if __name__ == "__main__":
    main(hparams)