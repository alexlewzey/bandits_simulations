#! python3
"""
plot pvalue over the number of iterations for a chi squared test with a different click through rate
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, chisquare, chi2

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.INFO,
    # filename='logs.txt'
)


class DataGenerator:
    """generates yes/no clicks for A and B groups"""

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def get_clicks(self) -> Tuple[float, float]:
        click1 = 1 if np.random.random() > self.p1 else 0
        click2 = 1 if np.random.random() > self.p2 else 0
        return (click1, click2)


def run_experiment(p1: float, p2: float, N: int):
    """calculates the pvalues for ab comparison of two success rates over p1 and p2 for N iterations"""
    pvalues = []
    contingency = np.ones(4).reshape(2, 2)  # rows A B, cols click noclick
    data_gen = DataGenerator(p1, p2)

    for i in range(N):
        click1, click2 = data_gen.get_clicks()
        contingency[0, click1] += 1
        contingency[1, click2] += 1

        trow = contingency.sum(0)
        tcol = contingency.sum(1)
        pcol = tcol / contingency.sum()
        exp = pcol.reshape(2, 1).dot(trow.reshape(1, 2))
        ch = ((contingency - exp) ** 2 / exp).sum()
        pval = 1 - chi2.cdf(ch, df=1)
        pvalues.append(pval)

    logger.info(f'\n{contingency}')
    return pvalues


@dataclass
class Spec:
    p1: float
    p2: float
    n: int


def plot_pvalues():
    params = [
        Spec(0.1, 0.1, 20_000),
        Spec(0.1, 0.105, 20_000),
        Spec(0.1, 0.11, 20_000),
        Spec(0.1, 0.12, 20_000),
    ]

    fig, ax = plt.subplots(len(params), figsize=(14, 8))
    for i, spec in enumerate(params):
        logger.info(spec)
        data = run_experiment(spec.p1, spec.p2, spec.n)
        logger.info(f'iter:{i} sum of p: {sum(data)}')
        ax[i].plot(data)
        ax[i].plot(np.ones(20_000) * 0.05)
        ax[i].set_title(spec)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_pvalues()
