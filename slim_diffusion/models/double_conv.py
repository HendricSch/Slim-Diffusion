import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Implementierung des DoubleConv Block, welcher in der U-Net Architektur verwendet wird. Besteht aus zwei Convolutional
    Layern mit GroupNorm und SiLU Aktivierungsfunktion. Besitzt einen Skip-Connection vom Input zum Output.
    """
    def __init__(self, channels):
        super(DoubleConv, self).__init__()

        self.first_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.first_norm = nn.GroupNorm(8, channels)

        self.second_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.second_norm = nn.GroupNorm(8, channels)

        self.activation = nn.SiLU()

    def forward(self, x):
        skip = x

        x = self.first_conv(x)
        x = self.first_norm(x)
        x = self.activation(x)

        x = self.second_conv(x)
        x = self.second_norm(x)
        x = self.activation(x + skip)

        return x


def main():

    batch_size = 1
    channels = 3
    img_size = 32

    x = torch.randn(batch_size, channels, img_size, img_size)

    model = DoubleConv(channels)
    print(model(x).shape)


if __name__ == '__main__':
    main()
