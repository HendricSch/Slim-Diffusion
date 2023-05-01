import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from slim_diffusion import sampling
from slim_diffusion import models
from slim_diffusion import datasets

from fid import FID

'''
Dieses Script wird für das Single-Batch-Overfitting verwendet. Es kann genutzt werden, um Hyperparameter des Modells zu
testen, ohne den gesamten Trainingsdatensatz zu trainieren.
'''

def main():
    # Setzen der Hyperparameter für das Single-Batch-Overfitting
    BATCH_SIZE = 256  # Batchgröße
    IMAGE_CHANNEL = 1  # Anzahl der Farbkanäle
    IMAGE_SIZE = 32  # Bildauflösung
    T = 200  # Anzahl der Diffusionsschritte T
    EPOCHS = 50  # Anzahl der Trainingsepochen
    STEPS_PER_EPOCH = 500  # Anzahl der Trainingsschritte pro Epoche
    LEARNING_RATE = 1e-3  # Lernrate
    DEVICE = "cuda"  # Device, auf dem das Training durchgeführt werden soll. "cuda" für GPU, "cpu" für CPU

    # Laden des Datensatzes und Erstellen des Dataloaders
    data = datasets.MNIST(img_size=IMAGE_SIZE)
    dataloader = data.get_dataloader(batch_size=BATCH_SIZE, shuffle=True)

    # Initialisieren des U-Net Modells
    model = models.UNet(data.channels, data.img_size, T, [16, 32, 64, 128]).to(DEVICE)

    # Initialisieren des Beta-Scheduler
    scheduler = sampling.LinearScheduler(T, device=DEVICE)

    # Initialisieren des Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialisieren des Lernraten-Schedulers
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS
    )

    # Initialisieren der FID Klasse für die spätere Berechnung der Frechet Inception Distanz
    fid = FID(feature=64, device="cuda")

    # Wählt eine zufällige Batch aus dem Datensatz aus. Diese wird für das Single-Batch-Overfitting verwendet.
    batch = next(iter(dataloader))[0]
    x_0 = batch.to(DEVICE)

    # Trainingsschleife
    for epoch in range(EPOCHS):

        # # Initialisieren des durchschnittlichen Fehlers. Dieser addiert den Fehler über alle Trainingsschritte.
        avg_loss = 0

        # Führt pro Epoche STEPS_PER_EPOCH Trainingsschritte durch
        for _ in tqdm(range(STEPS_PER_EPOCH)):
            # Erstellt für jedes Bild im Batch eine zufälliges t zwischen 1 und T
            t = torch.randint(1, T, (x_0.shape[0],)).to(DEVICE)

            # Führt für den kompletten Batch den Forward-Prozess durch. Es werden die Bilder x_t und die zugehörigen
            # Rauschvektoren epsilon zurückgegeben.
            x_t, epsilon = scheduler.forward_prozess(x_0, t)

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

            # Addiert den Fehler des aktuellen Trainingsschrittes auf den durchschnittlichen Fehler.
            avg_loss += loss.item()

            # Aktualisiert die Lernrate des Optimizers anhand des Lernraten-Schedulers.
            lr_scheduler.step()

        # Führt nach jeder Epoche einen Sampling-Prozess durch, um die Qualität der Samples zu überprüfen.
        sampler = sampling.Sampler(model, scheduler, device=DEVICE, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                                   channels=IMAGE_CHANNEL)

        sample = sampler.sample()
        sample = data.inverse_transform(sample[0].cpu().detach())

        plt.title(f"Epoch: {epoch} | Loss: {avg_loss / STEPS_PER_EPOCH * BATCH_SIZE}")
        plt.imshow(sample, cmap="gray")
        plt.show()

        # Berechnet die Frechet Inception Distanz zwischen den Samples und dem Trainingsdatensatz. Es wird für 10
        # zufällige Samples die FID berechnet und der Durchschnitt genommen.
        fid_scores = []

        for i in range(10):
            test_batch = next(iter(dataloader))[0]
            sample = sampler.sample()
            fid_scores.append(fid.compute(sample, test_batch))

        avg_fid = sum(fid_scores) / len(fid_scores)
        avg_loss = avg_loss / STEPS_PER_EPOCH * BATCH_SIZE

        print(f"Epoch: {epoch} | FID: {avg_fid} | Avg Loss: {avg_loss}")


if __name__ == '__main__':
    main()
