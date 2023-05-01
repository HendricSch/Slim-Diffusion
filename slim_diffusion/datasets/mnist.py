import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, ToPILImage
import os


class MNIST:
    """
    Diese Klasse bildet einen Wrapper um das MNIST Dataset. Sie stellt die Funktionen zur Verfügung, die für das
    Training von Slim Diffusion benötigt werden.
    """
    def __init__(self, img_size=32):
        self.img_size = img_size
        self.channels = 1

        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        # Preprocessing der Bilder. Wandelt Bilder mit dem Werteberich [0, 255] in Tensoren mit dem Wertebereich [-1, 1] um.
        self.transform = Compose([
            Resize(img_size),
            ToTensor(),
            Lambda(lambda x: (x * 2) - 1),
        ])

        # Inverse Transformation der Bilder. Wandelt Tensoren mit dem Wertebereich [-1, 1] in Bilder mit dem Wertebereich
        self._inverse_transform = Compose([
            Lambda(lambda x: (x + 1) / 2),
            Lambda(lambda x: x.permute(1, 2, 0)),
            Lambda(lambda x: x * 255.),
            Lambda(lambda x: x.numpy().astype(np.uint8)),
            ToPILImage(),
        ])

        # Lädt den MNIST Trainingsdatensatz herunter.
        self.training_data = datasets.MNIST(
            root=self.data_path,
            train=True,
            download=True,
            transform=self.transform
        )

        # Lädt den MNIST Testdatensatz herunter.
        self.test_data = datasets.MNIST(
            root=self.data_path,
            train=False,
            download=True,
            transform=self.transform
        )

        # Fügt die Trainings- und Testdaten zusammen.
        self.data = self.training_data + self.test_data

    def get_dataloader(self, batch_size=64, shuffle=True) -> DataLoader:
        """
        Erstellt einen DataLoader für den MNIST Datensatz.
        """
        return DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def inverse_transform(self, x: torch.Tensor) -> Image:
        """
        Wendet die inverse Transformation auf ein Bild an. Alle Werte über 1 werden auf 1 gesetzt und alle Werte unter -1
        werden auf -1 gesetzt. Dies muss gemacht werden, damit Matplotlib die Bilder korrekt darstellen kann.
        """
        y = x.clone()
        y[y > 1] = 1
        y[y < -1] = -1
        return self._inverse_transform(y)

    def inverse_transform_batch(self, x: torch.Tensor) -> list[Image]:
        """
        Wendet die inverse Transformation auf eine Batch von Bildern an.
        """
        return [self.inverse_transform(y) for y in x]


def main():
    image_size = 32
    batch_size = 64

    data = MNIST(
        img_size=image_size,
    )
    dataloader = data.get_dataloader(
        batch_size=batch_size,
    )

    for x, y in dataloader:
        print(x.shape)
        print(y.shape)
        break


if __name__ == '__main__':
    main()
