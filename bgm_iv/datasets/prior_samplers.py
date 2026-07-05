import numpy as np


class Gaussian_sampler:
    """Multivariate Gaussian sampler used for latent EGM warm starts."""

    def __init__(self, mean, sd=1, N=20000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size, len(self.mean)))
        self.X = self.X.astype("float32")

    def train(self, batch_size, label=False):
        del label
        indx = np.random.randint(low=0, high=self.total_size, size=batch_size)
        return self.X[indx, :]

    def get_batch(self, batch_size):
        return np.random.normal(
            self.mean,
            self.sd,
            (batch_size, len(self.mean)),
        ).astype("float32")

    def load_all(self):
        return self.X
