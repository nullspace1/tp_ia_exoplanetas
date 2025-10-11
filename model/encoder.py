import torch
import torch.nn as nn
import pytorch_tcn as TCN

class LightCurveEncoder(nn.Module):
    
    def __init__(self, input_size, output_dim):
        super(LightCurveEncoder, self).__init__()
        self.tcn = TCN.TCN(
            num_inputs=1,
            num_channels=[64, 64,64,64,64,64,64,64,64,64],
            kernel_size=3,
            dropout=0.1,
            causal=True,
            dilation_reset=512
        )
        self.pooling = nn.AdaptiveAvgPool1d(16)
        self.linear = nn.Linear(16 * 64, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
if __name__ == "__main__":
    input_size = 3000
    output_dim = 256
    model = LightCurveEncoder(input_size, output_dim)
    x = torch.randn(1, input_size)
    y = model(x)
    print(y.shape)  
    print(model.param_count())