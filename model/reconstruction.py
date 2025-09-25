import torch
import torch.nn as nn

from model.encoder import LightCurveEncoder
from model.decoder import LightCurveDecoder

class ReconstructionModel(nn.Module):
    def __init__(self,input_size,ouput_dim, base_channels=64):
        super().__init__()
        self.encoder = LightCurveEncoder(input_size, ouput_dim)
        self.decoder = LightCurveDecoder(ouput_dim, input_size, base_channels)

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed_x = self.decoder(latent_vector)
        return reconstructed_x
    
    
if __name__ == "__main__":
    B = 32
    T = 512
    input_size = 1
    latent_dim = 256

    model = ReconstructionModel(input_size=input_size, ouput_dim=latent_dim, base_channels=64)
    x = torch.randn(B, T, input_size)
    z = model(x)
    print(z.shape)