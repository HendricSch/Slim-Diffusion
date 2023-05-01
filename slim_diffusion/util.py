import torch

"""
Hilfsfunktionen für Slim Diffusion
"""


def grayscale_to_rgb(img_batch: torch.Tensor) -> torch.Tensor:
    """
    Konvertiert ein Tensor mit Bildern im Graustufenformat in ein Tensor mit Bildern im RGB-Format.
    """
    return torch.cat([img_batch, img_batch, img_batch], dim=1)


def dict_to_csv(data: dict, path: str) -> None:
    """
    Speichert den Inhalt eines Dictionaries in einer CSV-Datei.
    """
    with open(path, "w") as f:
        f.write(",".join(data.keys()) + "\n")

        for i in range(len(data[list(data.keys())[0]])):
            f.write(",".join([str(data[key][i]) for key in data.keys()]) + "\n")
