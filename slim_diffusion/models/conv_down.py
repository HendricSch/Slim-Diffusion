import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ConvDown(nn.Module):
    """
    Originale Implementierung des ConvDown Blocks.
    """

    # def __init__(self, in_channels, out_channels):
    #     super(ConvDown, self).__init__()
    #     self.conv = nn.Sequential(
    #         nn.MaxPool2d(2),
    #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    #         nn.SiLU(),
    #     )
    #
    # def forward(self, x):
    #     return self.conv(x)

    """
    Alternative Implementierung des ConvDown Blocks, welche ohne Pooling auskommt. Wurde benutzt, da die originale
    Implementierung teilweise zu CUDA Fehlern während des Trainings geführt hat.
    Quelle: https://github.com/huggingface/blog/blob/main/annotated-diffusion.md#:~:text=1)%2C%0A%20%20%20%20)%0A%0A%0Adef-,Downsample,-(dim%2C
    """
    def __init__(self, in_channels, out_channels):
        super(ConvDown, self).__init__()

        self.layer = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(in_channels * 4, out_channels, 1),
        )

    def forward(self, x):
        return self.layer(x)


def main():
    batch_size = 1
    in_channels = 64
    out_channels = 128
    img_size = 32

    x = torch.randn(batch_size, in_channels, img_size, img_size)

    model = ConvDown(in_channels, out_channels)
    print(model(x).shape)


if __name__ == '__main__':
    main()
