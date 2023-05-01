import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Test für das trainierbare TimeEmbedding. Für das U-Net wurde nicht dieser Wrapper verwendet, sondern direkt die
    PyTorch Embedding Klasse.
    """
    def __init__(self, steps, channels, img_size):
        super(TimeEmbedding, self).__init__()
        self.steps = steps
        self.channels = channels
        self.img_size = img_size

        self.embedding = nn.Embedding(steps, channels * img_size * img_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], self.channels, self.img_size, self.img_size)
        return x


def main():
    batch_size = 64
    channels = 3
    img_size = 32
    steps = 300

    batch = torch.randn(batch_size, channels, img_size, img_size)

    t = torch.randint(1, steps, (batch.shape[0],)).to("cuda")

    embedding = TimeEmbedding(steps, 8, img_size).to("cuda")

    print(f"batch shape:        {batch.shape}")
    print(f"t shape:            {t.shape}")
    print(f"embedding shape:    {embedding(t).shape}")
    print(f"result shape:       torch.Size([64, 8, 32, 32])")

    # print number of parameters for embedding
    print(f"number of parameters: {sum(p.numel() for p in embedding.parameters())}")


if __name__ == '__main__':
    main()
