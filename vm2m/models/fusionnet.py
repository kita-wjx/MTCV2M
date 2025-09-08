import torch
import torch.nn as nn
import math
from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, layer_idx, layers, dropout: float = 0.1, layer_norm_eps = 1e-12):
        super(FusionBlock, self).__init__()
        self.layer_idx = layer_idx
        self.layers = layers

        self.self_atten = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_output = nn.Sequential(OrderedDict([
            ("dense", nn.Linear(dim, dim)),
            ("LayerNorm", LayerNorm(dim, eps = layer_norm_eps)),
            ("dropout", nn.Dropout(dropout))
        ]))
        
        self.cross_atten = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_output = nn.Sequential(OrderedDict([
            ("dense", nn.Linear(dim, dim)),
            ("LayerNorm", LayerNorm(dim, eps = layer_norm_eps)),
            ("dropout", nn.Dropout(dropout))
        ]))
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim)),
            ('c_layer_norm', LayerNorm(dim, eps=layer_norm_eps)),
            ("dropout", nn.Dropout(dropout))
        ]))
        
        scale = dim ** -0.5
        self.alpha = nn.Parameter(scale * torch.randn(dim))
        if self.layer_idx != self.layers - 1:
            self.beta = nn.Parameter(scale * torch.randn(dim))
    
    def forward(self, hidden_states: torch.Tensor, x_image: torch.Tensor):
        # self attention
        hidden_states = self.self_atten(hidden_states, hidden_states, hidden_states)[0] # B*sec, 2, dim
        hidden_states = self.self_output(hidden_states)

        # query
        alpha = torch.sigmoid(self.alpha)
        query_hidden_states = alpha * hidden_states[:,0,:] + (1 - alpha) * hidden_states[:,1,:]
        query_hidden_states = query_hidden_states.unsqueeze(1) # B*sec, 1, dim
        
        # addition
        if self.layer_idx != self.layers - 1:
            beta = torch.sigmoid(self.beta)
            resdisual_hidden_states = beta * hidden_states[:,0,:] + (1 - beta) * hidden_states[:,1,:]
            resdisual_hidden_states = resdisual_hidden_states.unsqueeze(1) # B*sec, 1, dim
        
        # cross attention
        hidden_states = self.cross_atten(query_hidden_states, x_image, x_image)[0] # B*sec, 1, dim
        hidden_states = self.cross_output(hidden_states)
        
        # mlp
        hidden_states = self.mlp(hidden_states)
        
        # concat
        if self.layer_idx!= self.layers - 1:
            hidden_states = torch.cat([resdisual_hidden_states, hidden_states], dim=1) # B*sec, 2, dim
        
        return hidden_states

class FusionNet(nn.Module):
    def __init__(self, video_dim: int, dim: int, layers: int, heads: int, layer_norm_eps = 1e-12, hidden_dropout_prob = 0.1):
        super(FusionNet, self).__init__()
        self.video_dim = video_dim
        self.dim = dim
        
        # downsample
        self.downsample_image = nn.Conv2d(video_dim, dim, (2,2), 2)
        
        # video projection
        self.video_proj = nn.Linear(video_dim, dim)
        
        # fusion blocks
        self.layernorm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fusion_blocks = nn.ModuleList([FusionBlock(dim, heads, layer_idx, layers) for layer_idx in range(layers)])
        
        self.initialize_parameters(dim, layers)
    
    def initialize_parameters(self, width, layers):
        proj_std = (width ** -0.5) * ((2 * layers) ** -0.5)
        attn_std = width ** -0.5
        fc_std = (2 * width) ** -0.5
        for block in self.fusion_blocks:
            nn.init.normal_(block.cross_atten.in_proj_weight, std=attn_std)
            nn.init.normal_(block.cross_atten.out_proj.weight, std=proj_std)
            nn.init.normal_(block.cross_output.dense.weight, std=fc_std)
            
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            
            nn.init.normal_(block.self_atten.in_proj_weight, std=attn_std)
            nn.init.normal_(block.self_atten.out_proj.weight, std=proj_std)
            nn.init.normal_(block.self_output.dense.weight, std=fc_std)
        
    def forward(self, x_image, x_video, x_cls):
        assert x_image.size(0) == x_video.size(0) == x_cls.size(0)
        assert x_image.size(1) == x_video.size(1) == x_cls.size(1)
        assert x_image.size(-1) == x_video.size(-1) == self.video_dim
        assert x_cls.size(-1) == self.dim
        B, T, _ = x_video.size()
        _, _, P_image, C_image = x_image.size()
        
        # downsample video
        x_video = x_video.to(self.video_proj.weight)
        x_video = self.video_proj(x_video)
        
        # downsample image
        x_image = x_image.reshape(B*T, P_image, C_image).reshape(B*T, int(math.sqrt(P_image)), int(math.sqrt(P_image)), C_image).permute(0, 3, 1, 2).contiguous()
        x_image = self.downsample_image(x_image)
        x_image = x_image.permute(0, 2, 3, 1).contiguous().view(B, T, -1, self.dim)
    
        x_cat = torch.cat([x_video.unsqueeze(2), x_cls.unsqueeze(2)], dim=2) # B, sec, 2, dim
        hidden_states = self.layernorm(x_cat)
        hidden_states = self.dropout(hidden_states)
        
        # reshape
        hidden_states = hidden_states.reshape(B*T, 2, self.dim)
        x_image = x_image.reshape(B*T, -1, self.dim)
        
        for block in self.fusion_blocks:
            hidden_states = block(hidden_states, x_image)
        
        output = hidden_states.reshape(B, T, self.dim)
        
        return output

if __name__ == '__main__':
    net = FusionNet(1024, 768, 4, 8)
    x_image = torch.randn(4, 30, 576, 1024)
    x_video = torch.randn(4, 30, 1024)
    x_cls = torch.randn(4, 30, 768)
    x = net(x_image, x_video, x_cls)
