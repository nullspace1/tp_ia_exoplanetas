import torch
import torch.nn as nn

class LightCurveDecoder(nn.Module):
    def __init__(self, input_dim, output_length, base_channels=64):
        super(LightCurveDecoder, self).__init__()
        
        self.fc_layer = nn.Linear(input_dim, base_channels * output_length // 4)
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels=base_channels, out_channels=base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=base_channels // 2 , out_channels= base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=base_channels // 4, out_channels=1, kernel_size=3, padding=1),
        )


    def forward(self, latent_vector):
        x = self.fc_layer(latent_vector)
        x = x.view(x.size(0), 64, output_length // 4)
        x = self.net(x)
        x = x.squeeze(1)
        return x
    
    
if __name__ == "__main__":
    input_dim = 10
    output_length = 100
    model = LightCurveDecoder(input_dim, output_length)
    x = torch.randn(1, input_dim)
    y = model(x)
    print(y.shape)