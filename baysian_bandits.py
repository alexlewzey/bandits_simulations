import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.INFO,
    # filename='logs.txt'
)

NUM_TRAILS = 2000
BANDIT_PROBABILITIES = [0.2, 0.6, 0.7]


class Bandit:
    def __init__(self, p: float):
        self.p = p
        self.a = 1
        self.b = 1

    @property
    def params(self):
        return (self.a, self.b)

    def pull(self):
        return 1 if np.random.random() < self.p else 0

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x

    def __repr__(self):
        return f'Bandit(p={self.p}, a={self.a}, b={self.b})'


def plot_bandits(bandits: List[Bandit], trail_num: Optional[int] = None):
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 200)
    for bandit in bandits:
        y = stats.beta(bandit.a, bandit.b).pdf(x)
        ax.plot(x, y, label=f'Bandit: {bandit.p}')

    ax.set_title(f'Trail num: {trail_num}')
    plt.legend()
    plt.show()


def run_experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    for i in range(NUM_TRAILS):
        best_bandit: Bandit
        max_sample: float = -1
        all_samples: List = []
        for bandit in bandits:
            sample = bandit.sample()
            if sample > max_sample:
                max_sample = sample
                best_bandit = bandit
            all_samples.append(sample)
        try:
            if (i % 20 == 0):
                plot_bandits(bandits, i)
        except ZeroDivisionError:
            pass

        result_binary = best_bandit.pull()
        best_bandit.update(result_binary)
        logger.info(f'i={i}, samples={all_samples}')
        logger.info(f'bandit priors: {bandits}')


if __name__ == '__main__':
    run_experiment()
