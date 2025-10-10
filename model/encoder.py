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
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels = 64,out_channels = 128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels = 128,out_channels = output_dim,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.token_count = input_size // 2**4
        
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=1)
        
        cls_token = nn.Parameter(torch.randn(1, output_dim))
        self.cls_token = cls_token
        self.output_dim = output_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), self.output_dim, self.token_count)
        x = x.permute(0,2,1) ## (B, L, 256)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) ## (B, L, 256)
        x = torch.cat((cls_tokens, x), dim=1) ## (B, L+1, 256)
        x, _ = self.attention(x, x, x) ## (B, L+1, 256)
        x = x[:, 0]
        
        return x
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
if __name__ == "__main__":
    input_size = 3000
    output_dim = 512
    model = LightCurveEncoder(input_size, output_dim)
    x = torch.randn(1, input_size)
    y = model(x)
    print(y.shape)  
    print(model.param_count())