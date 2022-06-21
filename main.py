import os

import albumentations as A
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import torch
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models import resnet
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


def ResNet18(num_classes=2, pretrained=True):
    net = resnet.resnet18(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

class To3channel:
    """ Covert 1-channel image to 3-channel image by duplicating the channel.
    """
    def __call__(self, image):
        image = to_3channel(image)
        return image

def to_3channel(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
        image = np.concatenate([image]*3, axis=2)
        return np.ascontiguousarray(image)
    elif image.shape[-1] == 3:
        return image
    else:
        return np.ascontiguousarray(image[:,:,:3])


class LitMNIST(LightningModule):
    def __init__(self, data_dir="/home/junghoon/mnist"):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.transform = transforms.Compose(
            [
                To3channel(),
                transforms.ToTensor(),
            ]
        )
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.num_workers = 4

        # Define PyTorch model
        self.model = ResNet18(num_classes=self.num_classes, pretrained=True)

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = True
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = True
        )


if __name__ == "__main__":
    data_dir = "/home/junghoon/mnist"
    log_dir = os.path.join(data_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedir(log_dir)

    model = LitMNIST(data_dir=data_dir)
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=10,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir=log_dir),
    )
    trainer.fit(model)

    trainer.test()
