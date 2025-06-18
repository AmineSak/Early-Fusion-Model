import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class LitClassifier(L.LightningModule):
    def __init__(self, CLFModel):
        super().__init__()
        self.clf_model = CLFModel
        self.loss_fn = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.clf_model(x)
        loss = self.loss_fn(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
