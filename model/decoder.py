import torch
import torch.nn as nn

class LightCurveDecoder(nn.Module):
    def __init__(self, input_dim, output_length, base_channels=64):
        super(LightCurveDecoder, self).__init__()
         
        self.detection_head_1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        
        self.detection_head_2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        
        self.detection_head_3 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        
        self.fc_layer = nn.Linear(input_dim, base_channels * output_length // 4)
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels=base_channels, out_channels=base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=base_channels // 2 , out_channels= base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=base_channels // 4, out_channels=1, kernel_size=3, padding=1),
        )
        
        self.base = base_channels
        self.output_length = output_length


    def forward(self, latent_vector):
        x_1 = self.detection_head_1(latent_vector)
        x_2 = self.detection_head_2(latent_vector)
        x_3 = self.detection_head_3(latent_vector)
        
        x_1 = self.fc_layer(x_1)
        x_2 = self.fc_layer(x_2)
        x_3 = self.fc_layer(x_3)
        
        x_1 = x_1.view(x_1.size(0), self.base, self.output_length // 4)
        x_2 = x_2.view(x_2.size(0), self.base, self.output_length // 4)
        x_3 = x_3.view(x_3.size(0), self.base, self.output_length // 4)
        
        x_1 = self.net(x_1)
        x_2 = self.net(x_2)
        x_3 = self.net(x_3)
        
        return x_1.squeeze(1), x_2.squeeze(1), x_3.squeeze(1)
    
    
if __name__ == "__main__":
    input_dim = 10
    output_length = 100
    model = LightCurveDecoder(input_dim, output_length)
    x = torch.randn(1, input_dim)
    y1, y2, y3 = model(x)