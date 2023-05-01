import torch

from slim_diffusion.util import grayscale_to_rgb

from torchmetrics.image.fid import FrechetInceptionDistance  # Quelle: https://torchmetrics.readthedocs.io/en/latest/


def transform(img: torch.Tensor) -> torch.Tensor:
    """
    Transformiert einen Tensor in ein RGB-Bild, damit es von der FID-Metrik verarbeitet werden kann.
    """
    img = grayscale_to_rgb(img)
    img = (img + 1) / 2
    img = img * 255.
    img = img.type(torch.uint8)

    return img


class FID:
    """
    Wird für die Berechnung der Frechet Inception Distanz zwischen zwei Stichproben verwendet.
    """
    def __init__(self, feature=64, device="cuda"):
        self.feature = feature
        self.device = device

        self.fid = FrechetInceptionDistance(feature=feature).to(device)

    def compute(self, img_real, img_fake):
        with torch.no_grad():
            self.fid.reset()

            img_real = transform(img_real.to(self.device))
            img_fake = transform(img_fake.to(self.device))

            self.fid.update(img_real, real=True)
            self.fid.update(img_fake, real=False)

            score = self.fid.compute()

        return score

