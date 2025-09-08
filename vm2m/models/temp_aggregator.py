import torch.nn as nn

class TemporalAggregatorWithConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(TemporalAggregatorWithConv, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv1d(in_channels=input_channels, 
                              out_channels=output_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=kernel_size//2)
        self.gelu = nn.GELU()

    def forward(self, x):
        assert x.dim() == 3
        assert x.size(2) == self.input_channels
        x = x.transpose(1, 2)  # [B, C, S]
        x_aggregated = self.conv(x)  # [B, output_channels, S]
        x_aggregated = self.gelu(x_aggregated)
        return x_aggregated.mean(dim=2)  # pooling
