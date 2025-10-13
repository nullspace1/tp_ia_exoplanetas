
import torch.nn as nn
import pytorch_lightning as pl
import torch
import time
from pytorch_tcn import TCN
import torch.nn.functional as F

class ExoplanetCNNAttn(pl.LightningModule):
    def __init__(self, input_size, embedding_dim, period_bins, num_dilations, pool_size = 1):
        super().__init__()
        self.conv = TCN(
            num_inputs=1,
            num_channels=[embedding_dim] * num_dilations,
            kernel_size=1,
            dropout=0.1,
            dilations=[2**i for i in range(num_dilations)],
            use_norm="batch_norm"
        )
        
        self.pool = nn.AvgPool1d(pool_size, stride=pool_size)

        self.position_embedding = nn.Parameter(torch.randn(1, embedding_dim , input_size // pool_size) * 0.02, requires_grad=True)
        
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)
        
        self.fc = nn.Linear(embedding_dim, period_bins)
        
        self.attn_pool = nn.Linear(period_bins, 1)

        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = x + self.position_embedding
        x = x.permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = self.fc(x)
        scores = nn.Softmax(dim=1)(self.attn_pool(x))
        x = torch.sum(x * scores, dim=1)
        x = self.sigmoid(x) 
        return x

    def loss_fn(self, y_true, y_pred ):
        alpha=0.75
        gamma=2
        bce = nn.BCELoss()(y_pred, y_true)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * (1 - pt).pow(gamma) * bce - 0.05 * y_true.var(dim=0).mean()
        return loss.mean()

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("std_pred", torch.std(y_pred), prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("test_loss", loss,  prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("mean_pred", torch.mean(y_pred), prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    model = ExoplanetCNNAttn(3000, 32, 11,11)
    x = torch.randn(1, 3000)
    start_time = time.time()
    y = model(x)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(model.parameter_count())