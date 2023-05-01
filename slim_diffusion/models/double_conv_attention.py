import torch
import torch.nn as nn

from slim_diffusion.models.multihead_attention import MultiHeadAttention


class DoubleConvAttention(nn.Module):
    """
    Implementierung des DoubleConvAttention Block, welcher in der U-Net Architektur verwendet wird. Ist eine Erweiterung
    des DoubleConv Blocks, welcher zusätzlich eine MultiHeadAttention Schicht enthält.
    """
    def __init__(self, channels):
        super(DoubleConvAttention, self).__init__()

        self.first_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.first_norm = nn.GroupNorm(8, channels)

        self.second_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.second_norm = nn.GroupNorm(8, channels)

        self.multihead_attention = MultiHeadAttention(channels)

        self.activation = nn.SiLU()

    def forward(self, x):
        skip = x

        x = self.first_conv(x)
        x = self.first_norm(x)
        x = self.activation(x)

        x = self.multihead_attention(x)

        x = self.second_conv(x)
        x = self.second_norm(x)
        x = self.activation(x)

        x = self.activation(x + skip)

        return x


def main():
    batch_size = 1
    channels = 3
    img_size = 32

    x = torch.randn(batch_size, channels, img_size, img_size)

    model = DoubleConvAttention(channels)
    print(model(x).shape)


if __name__ == '__main__':
    main()
