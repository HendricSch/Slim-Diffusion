import torch
from matplotlib import pyplot as plt

from slim_diffusion import sampling
from slim_diffusion import models
from slim_diffusion import datasets

'''
Dieses Script kann genutzt werden, um ein Bild aus dem MNIST Datensatz zu sampeln. Dazu werden die Gewichte eines vorher
trainierten Modells geladen.
'''

def main():
    BATCH_SIZE = 1
    IMAGE_CHANNEL = 1
    IMAGE_SIZE = 32
    T = 200
    DEVICE = "cuda"

    data = datasets.MNIST(img_size=IMAGE_SIZE)
    model = models.UNet(data.channels, data.img_size, T, [16, 32, 64, 128]).to(DEVICE)
    model.load_state_dict(torch.load("weights/mnist.pt"))

    scheduler = sampling.LinearScheduler(T, device=DEVICE)

    sampler = sampling.Sampler(model, scheduler, device=DEVICE, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                               channels=IMAGE_CHANNEL)

    sample = sampler.sample()

    sample = data.inverse_transform_batch(sample.cpu().detach())

    plt.imshow(sample[0], cmap='gray')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
