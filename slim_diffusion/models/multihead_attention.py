import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Diese Klasse ist ein Wrapper für die MultiheadAttention Klasse von PyTorch, damit diese für das U-Net verwendet
    werden kann.
    """
    def __init__(self, channel, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.channel = channel
        self.num_heads = num_heads

        self.mha = torch.nn.MultiheadAttention(channel, 1, batch_first=True)

    def forward(self, x):
        b, c, w, h = x.shape

        x = x.flatten(2).permute(2, 0, 1)
        x = self.mha(x, x, x)[0]

        return x.permute(1, 2, 0).reshape(b, c, w, h)


def main():
    batch_size = 64
    channels = 1
    img_size = 32

    x = torch.randn(batch_size, channels, img_size, img_size)

    model = MultiHeadAttention(channels)
    print(model(x).shape)


if __name__ == '__main__':
    main()
