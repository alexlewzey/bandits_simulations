"""
create an animation of the Bayesian AB learning process for an arbitrary number of one armed bandits.
Global variables:
    NUM_ITERATIONS: number of times the animation will iterate ie select another one armed bandit and pull it
    BANDIT_PROBABILITIES: true win rate corresponding to each bandit, the Bayesian process incrementally discoverers
                            which one armed bandit is the best and exploits this knowledge, play around with this to see
                          different results.
"""
import logging
from typing import List, Tuple

import numpy as np
from scipy import stats
from baysian_bandits import Bandit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.INFO,
    # filename='logs.txt'
)

NUM_ITERATIONS: int = 2000
BANDIT_PROBABILITIES: List[float] = [0.5, 0.65, 0.7, 0.4]
bandit_names = ['A', 'B', 'C', 'D']

num_bandits: int = len(BANDIT_PROBABILITIES)
ylim = 10
fig, ax = plt.subplots()
ax.set_ylim(0, ylim)
ax.set_xlim(0, 1)
lines = [plt.plot([], [], label=f'Bandit: p={p}')[0] for p in BANDIT_PROBABILITIES]
plt.legend()


def init():
    """run at the start of animation cycle"""
    for line in lines:
        line.set_data([], [])
    return lines


def adjust_ylim(y, ylim) -> None:
    """if biggest y value is greater than ax ylim, it ylim to biggest y"""
    ymax = max(y) + 0.2
    if ymax > ylim:
        ax.set_ylim(0, ymax)


def animation(i):
    """run every new frame"""
    frame_params = plot_data[i]
    ax.set_title(f'Iteration: {i}')
    for j, line in enumerate(lines):
        a, b = frame_params[j]
        y = stats.beta(a, b).pdf(x)
        adjust_ylim(y, ylim)
        line.set_data(x, y)

    return lines


def generate_data() -> List[List[Tuple[int, int]]]:
    plot_parameters: List = []
    bandits = [Bandit(p, name=name) for p, name in zip(BANDIT_PROBABILITIES, bandit_names)]
    for i in range(NUM_ITERATIONS):
        best_bandit: Bandit
        best_sample: float = -1
        samples: List = []
        bandit_params: List = []
        for bandit in bandits:
            sample = bandit.sample()
            if sample > best_sample:
                best_sample = sample
                best_bandit = bandit
            samples.append(sample)
            bandit_params.append(bandit.params)

        result_binary = best_bandit.pull()
        best_bandit.update(result_binary)
        plot_parameters.append(bandit_params)
        logger.info(f'samples={samples}')
        logger.info(f'bandits={bandits}')
    return plot_parameters


x = np.linspace(0, 1, 200)
plot_data = generate_data()

anni = FuncAnimation(fig=fig, func=animation, frames=NUM_ITERATIONS, init_func=init, interval=50)
plt.show()
