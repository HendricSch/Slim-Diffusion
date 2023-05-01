import torch
import torch.nn as nn

from slim_diffusion.models.conv_down import ConvDown
from slim_diffusion.models.conv_up import ConvUp
from slim_diffusion.models.double_conv_attention import DoubleConvAttention


class UNetAttention(nn.Module):
    """
    Identisch zu der UNet Architektur, jedoch mit DoubleConvAttention Blöcken anstatt DoubleConv Blöcken.
    """
    def __init__(self, img_channel, img_size, steps, channel_list=None):
        super(UNetAttention, self).__init__()

        if channel_list is None:
            channel_list = [8, 16, 32, 64]

        self.img_channel = img_channel
        self.img_size = img_size
        self.steps = steps
        self.channel_list = channel_list
        self.depth = len(channel_list)

        self.first_conv = nn.Sequential(
            nn.Conv2d(img_channel, self.channel_list[0], kernel_size=3, padding=1),
            nn.GroupNorm(1, self.channel_list[0]),
            nn.SiLU(),
        )

        self.encoder_double_conv = nn.ModuleList([
            DoubleConvAttention(self.channel_list[i]) for i in range(self.depth)
        ])

        self.encoder_conv_down = nn.ModuleList([
            ConvDown(self.channel_list[i], self.channel_list[i + 1]) for i in range(self.depth - 1)
        ])

        self.decoder_double_conv = nn.ModuleList([
            DoubleConvAttention(self.channel_list[i]) for i in range(self.depth - 2, -1, -1)
        ])

        self.decoder_conv_up = nn.ModuleList([
            ConvUp(self.channel_list[i], self.channel_list[i - 1]) for i in range(self.depth - 1, 0, -1)
        ])

        self.bottleneck = DoubleConvAttention(self.channel_list[-1])

        self.last_conv = nn.Sequential(
            nn.Conv2d(self.channel_list[0], img_channel, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(img_channel, img_channel, kernel_size=3, padding=1),
        )

        self.time_embendding = nn.Embedding(steps, 1)

    def forward(self, x, t,):

        skip_connections = []

        x = self.first_conv(x)

        for i in range(self.depth - 1):
            b, c, h, w = x.shape
            t_emb = self.time_embendding(t).view(b, 1, 1, 1).repeat(1, c, h, w)

            x = self.encoder_double_conv[i](x + t_emb)
            skip_connections.append(x)
            x = self.encoder_conv_down[i](x)

        x = self.bottleneck(x)

        for i in range(self.depth - 1):
            x = self.decoder_conv_up[i](x)
            x = self.decoder_double_conv[i](x + skip_connections.pop())

        x = self.last_conv(x)

        return x


def main():
    batch_size = 64
    channel = 3
    img_size = 32
    steps = 1000
    device = "cuda"

    model = UNetAttention(channel, img_size, steps, channel_list=[16, 32, 64, 128]).to(device)

    x = torch.randn(batch_size, channel, img_size, img_size).to(device)
    t = torch.randint(1, steps, (x.shape[0],)).to(device)

    # print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # print model output shape
    print(model(x, t).shape)


if __name__ == '__main__':
    main()
