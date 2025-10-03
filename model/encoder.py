import torch
import torch.nn as nn


class LightCurveEncoder(nn.Module):
    
    def __init__(self, input_size, output_dim):
        super(LightCurveEncoder, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(in_channels = 32,out_channels = 64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(in_channels = 64,out_channels = 128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(in_channels = 128,out_channels = 256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8)
        )
        
        self.token_count = input_size // 8**4
        
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=1)
        
        cls_token = nn.Parameter(torch.randn(1, 256))
        self.cls_token = cls_token

        
        
        
    def forward(self, x):
        x = self.conv_layer(x) ## (B, 256 * L)
        x = x.view(x.size(0),256, self.token_count) ## (B, 256, L)
        x = x.permute(0,2,1) ## (B, L, 256)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) ## (B, L, 256)
        x = torch.cat((cls_tokens, x), dim=1) ## (B, L+1, 256)
        x, _ = self.attention(x, x, x) ## (B, L+1, 256)
        x = x[0] ## (B, 256)
        
        return x
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
if __name__ == "__main__":
    input_size = 65000
    output_dim = 100
    model = LightCurveEncoder(input_size, output_dim)
    x = torch.randn(1, 1, input_size)
    y = model(x)
    print(y.shape)  
    print(model.param_count())