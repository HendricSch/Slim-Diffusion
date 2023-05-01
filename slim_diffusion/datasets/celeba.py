import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, ToPILImage
import os
import torch
from PIL.Image import Image


class CelebADataset(Dataset):
    """
    Erstellt eine Dataset Klasse für den CelebA Datensatz, damit dieser für das Training von Slim Diffusion verwendet
    werden kann. Die Bilder wurden unter der folgenden URL heruntergeladen: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
    Die Bilder wurden auf eine Größe von 32x32 Pixeln herunterskaliert und als Tensor gespeichert. Dieser Tensor wird
    beim Initialisieren der Klasse geladen. Da dieser Tensor eine Größe von 2.43 GB hat, kann dieser nicht auf GitHub
    hochgeladen werden.
    """

    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "celeba_dataset_32px.pt")
        self.data_tensor = torch.load(self.data_path)

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx]


class CelebA:
    """
    Diese Klasse bildet einen Wrapper um die CelebADataset Klasse. Sie stellt die Funktionen zur Verfügung, die für das
    Training von Slim Diffusion benötigt werden.
    """
    def __init__(self):
        self.channels = 3
        self.img_size = 32

        # Preprocessing der Bilder. Wandelt Bilder mit dem Werteberich [0, 255] in Tensoren mit dem Wertebereich [-1, 1] um.
        self.transform = Compose([
            Resize(self.img_size),
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

        self.dataset = CelebADataset()

        self.data = self.dataset.data_tensor

    def get_dataloader(self, batch_size, shuffle=True):
        """
        Erstellt einen DataLoader für den CelebA Datensatz.
        """
        return DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def inverse_transform(self, x):
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
    batch_size = 64

    data = CelebA()

    dataloader = data.get_dataloader(
        batch_size=batch_size
    )

    for x in dataloader:
        print(x.shape)
        break


if __name__ == '__main__':
    main()
