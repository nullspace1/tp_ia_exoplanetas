from encoder import LightCurveEncoder
from processor import ParameterEstimator
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, embedding_dim, period_bins):
        super(Model, self).__init__()
        self.encoder = LightCurveEncoder(input_size, embedding_dim)
        self.processor = ParameterEstimator(embedding_dim, period_bins)

    def forward(self, x):
        x = self.encoder(x)
        period = self.processor(x)
        return period
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
if __name__ == "__main__":
    input_size = 1500
    embedding_dim = 100
    period_bins = 100
    model = Model(input_size, embedding_dim, period_bins)
    print(model.param_count())