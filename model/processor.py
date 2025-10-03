
import torch.nn as nn
import torch
class ParameterEstimator(nn.Module):
    
    def __init__(self, embedding_dim, period_bins, duration_bins):
        
        super(ParameterEstimator, self).__init__()
        self.embedding_dim = embedding_dim
        self.period_bins = period_bins
        self.duration_bins = duration_bins
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, period_bins)
        self.fc3 = nn.Linear(128, duration_bins)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        period = self.sigmoid(self.fc2(x))
        duration = self.sigmoid(self.fc3(x))
        return period, duration
    
    
if __name__ == "__main__":
    model = ParameterEstimator(100, 100, 100)
    x = torch.randn(1, 100)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)