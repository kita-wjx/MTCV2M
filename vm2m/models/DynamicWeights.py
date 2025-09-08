import torch
import torch.nn as nn

class DWNet(nn.Module):
    def __init__(self, dim, num_blocks, patch_length):
        super(DWNet, self).__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.patch_length = patch_length
        
        self.blocks = nn.ModuleList([
            DWNetBlock(dim)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, C, T)
        assert x.shape[-1] % self.patch_length == 0
        x = x.reshape(x.shape[0], x.shape[1], x.shape[-1] // self.patch_length, self.patch_length)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x
        
class DWNetBlock(nn.Module):
    def __init__(self, dim):
        super(DWNetBlock, self).__init__()
        self.inter_pool = nn.AdaptiveAvgPool1d(1)
        
        self.intra_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(dim)
        
        self.inter_conv = nn.Conv1d(dim, dim, kernel_size=1)
        

    def forward(self, x):
        # Intra-patch features
        assert x.dim() == 4
        _, _, T, _ = x.shape

        intra_features = nn.AdaptiveAvgPool2d((T, 1))(x)
        intra_features = intra_features.squeeze(-1)
        
        intra_features = self.intra_conv(intra_features)
        
        intra_features = self.batch_norm(intra_features)
        intra_weight = nn.ReLU()(intra_features)
        
        inter_features = self.inter_pool(intra_weight)
        inter_features = self.inter_conv(inter_features)
        inter_weight = nn.ReLU()(inter_features)
        
        weight = intra_weight + inter_weight
        weight = weight.unsqueeze(-1)
        x = weight * x
 
        return x

if __name__ == '__main__':
    model = DWNet(dim=1024, num_blocks=3, patch_length=50)
    x = torch.randn(3, 1500, 1024)
    y = model(x)
    print(y.shape)
