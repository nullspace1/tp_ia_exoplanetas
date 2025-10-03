
import torch.nn as nn
import torch
class ParameterEstimator(nn.Module):
    
    def __init__(self, embedding_dim, period_bins):
        
        super(ParameterEstimator, self).__init__()
        self.embedding_dim = embedding_dim
        self.period_bins = period_bins
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, period_bins)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        period = self.sigmoid(self.fc2(x))
        return period
    
    
if __name__ == "__main__":
    model = ParameterEstimator(100, 100)
    x = torch.randn(1, 100)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)