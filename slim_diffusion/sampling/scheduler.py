from abc import ABC, abstractmethod
import torch
import math


class Scheduler(ABC):
    """
    Abstrakte Klasse für den Beta-Scheduler. Führt den Forward-Prozess durch.
    """
    def __init__(self, T, device):
        self.T = T
        self.device = device

        # Tensor mit den Beta und Alpha Werten
        self.beta_tensor = None
        self.alpha_tensor = None

        # Zwischenwerte für den Forward-Prozess. Werden mit der Methode _pre_calc() berechnet.
        self.kumprodukt_alpha_tensor = None
        self.kehr_wurz_alpha = None
        self.kumprodukt_wurz_alpha_tensor = None
        self.kumprodukt_wurz_alpha_shifted = None

        # Initialisierung der Beta Werte
        self._init_beta()

        # Berechnung der Zwischenwerte
        self._pre_calc()

    @abstractmethod
    def _init_beta(self):
        """
        Initialisiert die Beta Werte. Wird in den Unterklassen implementiert.
        """
        pass

    def expand_tensor_shape(self, value, t, shape_tensor):
        """
        Hilfsfunktion, um die Dimensionen eines Tensors zu erweitern. Wird für den Forward-Prozess benötigt, um direkt
        einen ganzen Batch zu verarbeiten.
        """
        gathered = value.gather(-1, t)
        batch_size = t.shape[0]
        gathered = gathered.reshape(batch_size, *((1,) * (len(shape_tensor) - 1)))
        return gathered.to(t.device)

    def _pre_calc(self):
        """
        Berechnet die Zwischenwerte für den Forward-Prozess.
        """
        self.alpha_tensor = 1. - self.beta_tensor
        self.kumprodukt_alpha_tensor = torch.cumprod(self.alpha_tensor, dim=0)
        self.kehr_wurz_alpha = torch.sqrt(1.0 / self.alpha_tensor)
        self.kumprodukt_wurz_alpha_tensor = torch.sqrt(self.kumprodukt_alpha_tensor)
        self.kumprodukt_wurz_alpha_shifted = torch.sqrt(1. - self.kumprodukt_alpha_tensor)

    def forward_prozess(self, batch, t):
        """
        Führt den Forward-Prozess durch. Berechnet aus einem Batch von Bildern und einem Zeitpunkt t das entsprechende
        Bild x_t und das Rauschen epsilon.
        """

        # Wählt einen zufälligen Raumvektor epsilon aus der Normalverteilung
        epsilon = torch.randn_like(batch, device=self.device)

        # Berechnte mit der geschlossenen Form des Forward-Prozesses das Bild x_t für einen kompletten Batch
        mittelwert = self.expand_tensor_shape(self.kumprodukt_wurz_alpha_tensor, t, batch.shape)
        varianz = self.expand_tensor_shape(self.kumprodukt_wurz_alpha_shifted, t, batch.shape)
        x_t = mittelwert * batch + varianz * epsilon

        return x_t, epsilon


class ConstantScheduler(Scheduler):
    """
    Konstanter Beta-Scheduler.
    """
    def __init__(self, T, device, noise=0.002):
        self.noise = noise

        super().__init__(T, device)

    def _init_beta(self):
        self.beta_tensor = torch.ones(self.T, device=self.device) * self.noise


class LinearScheduler(Scheduler):
    """
    Linearer Beta-Scheduler.
    """
    def __init__(self, T, device, start=0.0001, end=0.02):
        self.start = start
        self.end = end
        super().__init__(T, device)

    def _init_beta(self):
        t = torch.arange(0, self.T)
        self.beta_tensor = self.start + (self.end - self.start) * t / self.T
        self.beta_tensor = self.beta_tensor.to(self.device)


class QuadratScheduler(Scheduler):
    """
    Quadratischer Beta-Scheduler.
    """
    def __init__(self, T, device, start=0.0001, end=0.01):
        self.start = start
        self.end = end

        super().__init__(T, device)

    def _init_beta(self):
        t = torch.arange(0, self.T)
        self.beta_tensor = (math.sqrt(self.start) + (math.sqrt(self.end) - math.sqrt(self.start)) * t / self.T) ** 2
        self.beta_tensor = self.beta_tensor.to(self.device)


class SigmoideScheduler(Scheduler):
    """
    Sigmoid Beta-Scheduler.
    """
    def __init__(self, T, device, start=0.0001, end=0.005):
        self.start = start
        self.end = end

        super().__init__(T, device)

    def _init_beta(self):
        t = torch.arange(0, self.T)
        t_cap = 12 * t / self.T - 6
        self.beta_tensor = torch.sigmoid(t_cap) * (self.end - self.start) + self.start
        self.beta_tensor = self.beta_tensor.to(self.device)
