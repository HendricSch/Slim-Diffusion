import torch
import math

from slim_diffusion.sampling.scheduler import Scheduler, LinearScheduler
from slim_diffusion.models.unet import UNet


class Sampler:
    """
    Diese Klasse wird für den Sampling-Prozess von Slim Diffusion verwendet.
    """
    def __init__(self, model, scheduler: Scheduler, device, batch_size=4, image_size=32, channels=3):
        self.model = model # trainiertes Modell
        self.beta_scheduler = scheduler # Beta-Scheduler
        self.device = device # Device, welches für das Sampling verwendet werden soll. (GPU oder CPU)
        self.batch_size = batch_size # Batch-Size eines Samples
        self.image_size = image_size # Bildgröße eines Samples (Wird durch den Trainingsdatensatz festgelegt)
        self.channels = channels # Anzahl der Farbkanäle eines Samples (Wird durch den Trainingsdatensatz festgelegt)

        self.T = scheduler.T
        self.alpha_tensor = scheduler.alpha_tensor
        self.beta_tensor = scheduler.beta_tensor

    @torch.no_grad()
    def sample_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Führt einen einzelnen Sampling-Schritt durch.
        """
        alpha_t_produkt = torch.prod(self.alpha_tensor[:t])
        t_tensor = torch.ones(x_t.shape[0], dtype=torch.int64, device=self.device) * t

        skalierungsfaktor = (1 / math.sqrt(self.alpha_tensor[t]))
        beta_t = self.beta_tensor[t]
        epsilon_pred = self.model(x_t, t_tensor)

        x_t_min_1 = skalierungsfaktor * (x_t - beta_t * epsilon_pred / math.sqrt(1 - alpha_t_produkt))

        if t > 1:
            # Falls t > 1, wird ein zufälliger Rauschvektor z erzeugt und auf x_t_min_1 addiert.
            z = torch.randn_like(x_t_min_1)
            skalierungsfaktor_z = torch.prod(self.alpha_tensor[:t - 1])
            z_param = self.beta_tensor[t] * math.sqrt(1 - skalierungsfaktor_z) / math.sqrt(1 - alpha_t_produkt)

            x_t_min_1 = x_t_min_1 + z * z_param

        return x_t_min_1

    @torch.no_grad()
    def sample(self):
        """
        Führt den Sampling-Prozess durch. (T Schritte)
        """
        sample_shape = (self.batch_size, self.channels, self.image_size, self.image_size)
        x_T = torch.randn(sample_shape).to(self.device)

        for i in range(self.T - 1, 0, -1):
            x_T = self.sample_step(x_T, i)

        return x_T


def main():
    scheduler = LinearScheduler(T=100, device="cuda")
    model = UNet(3, 32, 100, channel_list=[32, 64, 128, 256, 512, 1024]).to("cuda")
    sampler = Sampler(model, scheduler, "cuda")

    print(sampler.sample().shape)


if __name__ == '__main__':
    main()
