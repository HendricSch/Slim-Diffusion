import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from slim_diffusion import sampling
from slim_diffusion import models
from slim_diffusion import datasets

'''
Dieses Script wird für das Training von Slim Diffusion verwendet. Es enthält die Trainingsschleife, welche genutzt wird,
um das neu initialisierte Modell zu trainieren.
'''


def main():

    # Setzen der Hyperparameter für das Training
    BATCH_SIZE = 256  # Batchgröße
    IMAGE_CHANNEL = 3  # Anzahl der Farbkanäle
    IMAGE_SIZE = 32  # Bildauflösung
    T = 200  # Anzahl der Diffusionsschritte T
    EPOCHS = 10  # Anzahl der Trainingsepochen
    LEARNING_RATE = 2e-4  # Lernrate
    DEVICE = "cuda"  # Device, auf dem das Training durchgeführt werden soll. "cuda" für GPU, "cpu" für CPU

    # Laden des Datensatzes und Erstellen des Dataloaders
    data = datasets.CelebA()
    dataloader = data.get_dataloader(batch_size=BATCH_SIZE)

    # Initialisieren des U-Net Modells
    model = models.UNet(data.channels, data.img_size, T, [32, 64, 128, 256, 256]).to(DEVICE)

    # Initialisieren des Beta-Scheduler
    beta_scheduler = sampling.LinearScheduler(T, device=DEVICE)

    # Initialisieren des Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialisieren des Lernraten-Schedulers
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(dataloader),
        epochs=EPOCHS,
    )

    # Trainingsschleife, welche für jede Epoche durch den gesamten Datensatz iteriert
    for epoch in tqdm(range(EPOCHS)):

        # Initialisieren des durchschnittlichen Fehlers. Dieser addiert den Fehler über alle Batches und wird am Ende
        # der Epoche durch die Anzahl der Batches geteilt, um den durchschnittlichen Fehler zu erhalten.
        avg_loss = 0

        # Iterieren über alle Batches im Trainingsdatensatz
        for i, x_0, in enumerate(dataloader):
            # Lade die Daten auf das Device
            x_0 = x_0.to(DEVICE)

            # Erstellt für jedes Bild im Batch eine zufälliges t zwischen 1 und T
            t = torch.randint(1, T, (x_0.shape[0],)).to(DEVICE)

            # Führt für den kompletten Batch den Forward-Prozess durch. Es werden die Bilder x_t und die zugehörigen
            # Rauschvektoren epsilon zurückgegeben.
            x_t, epsilon = beta_scheduler.forward_prozess(x_0, t)

            # Berechnet die Vorhersage des Modells. Für jedes Bild x_t im Batch wird ein Rauschvektor epsilon_pred
            # vorhergesagt.
            epsilon_pred = model(x_t, t)

            # Berechnet den Fehler zwischen den vorhergesagten Rauschvektoren epsilon_pred und den tatsächlichen
            # Rauschvektoren epsilon.
            loss = F.mse_loss(epsilon_pred, epsilon)

            # Berechnet die Gradienten und führt einen Optimierungsschritt durch.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Addiert den Fehler des aktuellen Batches auf den durchschnittlichen Fehler.
            avg_loss += loss.item()

            # Aktualisiert die Lernrate des Optimizers anhand des Lernraten-Schedulers.
            lr_scheduler.step()

        print(f"Epoch: {epoch} | Avg Loss: {avg_loss / len(dataloader)}")

        # Führt nach jeder Epoche einen Sampling-Prozess durch, um die Qualität der Samples zu überprüfen.
        sampler = sampling.Sampler(model, beta_scheduler, device=DEVICE, batch_size=4, image_size=IMAGE_SIZE,
                                   channels=IMAGE_CHANNEL)

        sample = sampler.sample()
        sample = data.inverse_transform_batch(sample.cpu().detach())

        plt.figure(figsize=(10, 10))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(sample[i], cmap="gray")
            plt.axis("off")
        plt.suptitle(f"Epoch: {epoch} | Loss: {avg_loss / len(dataloader)}")
        plt.show()

    # Speichert die Gewichte des Modells
    torch.save(model.state_dict(), f"weights/test.pt")


if __name__ == '__main__':
    main()
