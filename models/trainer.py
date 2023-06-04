from pytorch_lightning import LightningModule
from model import TimeSeriesLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import GitHubCollabDataset, GithubCollabTestDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import F1Score


# set up trainer for training the task is binary classification
class GithubCollabTrainer(LightningModule):
    def __init__(self, hparams):
        super(GithubCollabTrainer, self).__init__()
        self.save_hyperparameters()
        self.param = hparams
        self.model = TimeSeriesLSTM(
            input_size=hparams.input_size,
            hidden_size=hparams.hidden_size,
            num_layers=hparams.num_layers,
        )
        self.train_f1 = F1Score(task="binary", num_classes=2)
        self.val_f1 = F1Score(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        self.train_f1(pred, y)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        self.log("train_f1_epoch", self.train_f1.compute())
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        self.val_f1(pred, y)
        self.log("val_f1", self.val_f1, on_step=True, on_epoch=False)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_f1_epoch", self.val_f1.compute())
        print("Validation f1: ", self.val_f1.compute().item())
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.param.learning_rate,
            weight_decay=self.param.weight_decay,
        )
        return optimizer

    def train_dataloader(self):
        train_dataset = GitHubCollabDataset(
            seq_len=self.param.seq_len,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.param.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        val_dataset = GithubCollabTestDataset(
            seq_len=self.param.seq_len,
        )
        return DataLoader(
            val_dataset,
            batch_size=self.param.batch_size,
            shuffle=False,
        )
