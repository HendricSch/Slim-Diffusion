import torch
import torch.nn as nn


class ConvUp(nn.Module):

    """
    Originale Implementierung des ConvUp Blocks.
    """
    # def __init__(self, in_channels, out_channels):
    #     super(ConvUp, self).__init__()
    #
    #     self.conv = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
    #         nn.GroupNorm(8, out_channels),
    #         nn.SiLU(),
    #     )
    #
    # def forward(self, x):
    #     return self.conv(x)

    """
    Alternative Implementierung des ConvUp Blocks, welche durch die Verwendung von Upsampling ohne Transpose Convolution
    auskommt. Wurde verwendet, um das Training des CIFAR10 Datensatzes stabiler zu machen, doch hat sich nicht als
    effektiver erwiesen.
    Quelle: https://github.com/huggingface/blog/blob/main/annotated-diffusion.md#:~:text=%2B%20x%0A%0A%0Adef-,Upsample,-(dim%2C
    """
    def __init__(self, in_channels, out_channels):
        super(ConvUp, self).__init__()

        self.dim = in_channels
        self.out_dim = out_channels

        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.dim, self.out_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layer(x)


def main():

    batch_size = 1
    in_channels = 64
    out_channels = 32
    img_size = 32

    x = torch.randn(batch_size, in_channels, img_size, img_size)

    model = ConvUp(in_channels, out_channels)
    print(model(x).shape)


if __name__ == '__main__':
    main()
