from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, Metric

###############################
#    Multilayer Perceptron    #
###############################

IMAGE_SIZE = 28


class CNN(pl.LightningModule):
    """
    Convolutional Neural Network (CNN) to solve MNIST with PyTorch Lightning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 10,
        metric: type[Metric] = Accuracy,
        lr_rate: float = 0.001,
        seed: Optional[int] = None,
    ):
        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        if out_channels == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=out_channels)
        self.lr_rate = lr_rate

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding="same",
        )
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            padding="same",
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = nn.Linear(7 * 7 * 64, 2048)
        self.l2 = nn.Linear(2048, out_channels)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        input_layer = x.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        return logits

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_id: int
    ) -> torch.Tensor:
        """ """
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_id: int
    ) -> torch.Tensor:
        """ """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor:
        """ """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
        return loss
