
import torch.nn as nn
import pytorch_lightning as pl
import torch

from model.encoder import LightCurveEncoder
from model.processor import ParameterEstimator


class Model(pl.LightningModule):
    def __init__(self, input_size, embedding_dim, period_bins):
        super().__init__()
        self.encoder = LightCurveEncoder(input_size, embedding_dim)
        self.processor = ParameterEstimator(embedding_dim, period_bins)

    def forward(self, x):
        x = self.encoder(x)
        period = self.processor(x)
        return period
    
    def loss_fn(self, y_true, y_pred):
        return nn.BCELoss()(y_pred, y_true)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("train_loss", loss, on_step=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("test_loss", loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("val_loss", loss, on_step=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    model = Model(3000, 256, 100)
    x = torch.randn(1, 3000)
    y = model(x)
    print(y.shape)
    print(model.parameter_count())