import torch
import torch.nn as nn


class LightCurveEncoder(nn.Module):
    
    def __init__(self, input_size, output_dim):
        super(LightCurveEncoder, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels = 32,out_channels = 64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc_layer = nn.Linear(64*input_size // 4, output_dim)
        
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
    
if __name__ == "__main__":
    input_size = 100
    output_dim = 10
    model = LightCurveEncoder(input_size, output_dim)
    x = torch.randn(1, 1, input_size)
    y = model(x)
    print(y.shape)  