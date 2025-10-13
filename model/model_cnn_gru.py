import torch.nn as nn
import pytorch_lightning as pl
import torch
import time

class ExoplanetCNNGRU(pl.LightningModule):
    def __init__(self, input_size=3000, num_classes=11, hidden_size=128):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        
        self.conv8 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()
        
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = x.permute(0, 2, 1)
        
        gru_out, _ = self.gru(x)
        
        x = gru_out[:, -1, :]
        
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

    def loss_fn(self, y_true, y_pred):
        alpha = 0.75
        gamma = 2
        
        bce = nn.BCELoss()(y_pred, y_true)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * (1 - pt).pow(gamma) * bce
        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y, y_pred)
        self.log("test_loss", loss, prog_bar=True, logger=True)
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
    model = ExoplanetCNNGRU(input_size=3000, num_classes=100, hidden_size=128)
    x = torch.randn(1, 3000)
    start_time = time.time()
    y = model(x)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Output shape: {y.shape}")
