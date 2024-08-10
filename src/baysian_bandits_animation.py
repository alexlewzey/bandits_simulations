"""Create an animation of the Bayesian AB learning process for an arbitrary number of
one armed bandits.

Global variables:
    NUM_ITERATIONS: number of times the animation will iterate ie select another one
        armed bandit and pull it
    BANDIT_PROBABILITIES: true win rate corresponding to each bandit, the Bayesian
        process incrementally discoverers which one armed bandit is the best and
        exploits this knowledge, play around with this to see
        different results.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import stats

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    level=logging.INFO,
)
tmp_dir = Path(__file__).parent.parent / "tmp"
tmp_dir.mkdir(exist_ok=True)


NUM_ITERATIONS: int = 500
BANDIT_PROBABILITIES: List[float] = [0.5, 0.65, 0.7, 0.4]
bandit_names = ["A", "B", "C", "D"]
# PCT_RANDOM: float = 0.0
# ratio = int(round(1 / PCT_RANDOM))

NUM_TRAILS = 2000
BANDIT_PROBABILITIES = [0.2, 0.6, 0.7]


class Bandit:
    def __init__(self, p: float, name: str):
        self.name = name
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
        return f"Bandit(name={self.name}, p={self.p}, a={self.a}, b={self.b})"


def plot_bandits(bandits: List[Bandit], trail_num: Optional[int] = None):
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 200)
    for bandit in bandits:
        y = stats.beta(bandit.a, bandit.b).pdf(x)
        ax.plot(x, y, label=f"Bandit: {bandit.p}")

    ax.set_title(f"Trail num: {trail_num}")
    plt.legend()
    plt.show()


def run_experiment():
    bandits = [Bandit(p, str(p)) for p in BANDIT_PROBABILITIES]

    for i in range(NUM_TRAILS):
        best_bandit: Optional[Bandit] = None
        max_sample: float = -1
        all_samples: List = []
        for bandit in bandits:
            sample = bandit.sample()
            if sample > max_sample:
                max_sample = sample
                best_bandit = bandit
            all_samples.append(sample)
        try:
            if i % 20 == 0:
                plot_bandits(bandits, i)
        except ZeroDivisionError:
            pass

        result_binary = best_bandit.pull()
        best_bandit.update(result_binary)
        logger.info(f"i={i}, samples={all_samples}")
        logger.info(f"bandit priors: {bandits}")


num_bandits: int = len(BANDIT_PROBABILITIES)
ylim = 10
fig, ax = plt.subplots()
ax.set_ylim(0, ylim)
ax.set_xlim(0, 1)
lines = [plt.plot([], [], label=f"Bandit: p={p}")[0] for p in BANDIT_PROBABILITIES]
plt.legend()


def init():
    """Run at the start of animation cycle."""
    for line in lines:
        line.set_data([], [])
    return lines


def adjust_ylim(y, ylim) -> None:
    """If biggest y value is greater than ax ylim, it ylim to biggest y."""
    ymax = max(y) + 0.2
    if ymax > ylim:
        ax.set_ylim(0, ymax)


def animation(i):
    """Run every new frame."""
    frame_params = plot_data[i]
    ax.set_title(f"Iteration: {i}")
    for j, line in enumerate(lines):
        a, b = frame_params[j]
        y = stats.beta(a, b).pdf(x)
        adjust_ylim(y, ylim)
        line.set_data(x, y)

    return lines


def generate_data() -> List[List[Tuple[int, int]]]:
    plot_parameters: List = []
    bandits = [
        Bandit(p, name=name) for p, name in zip(BANDIT_PROBABILITIES, bandit_names)
    ]
    best_bandit: Optional[Bandit] = None
    for _ in range(NUM_ITERATIONS):
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

        # if i % ratio == 0:
        #     print(f'selecting random sample: i={i}')
        #     best_bandit = random.choice(bandits)

        result_binary = best_bandit.pull()
        best_bandit.update(result_binary)
        plot_parameters.append(bandit_params)
        logger.info(f"samples={samples}")
        logger.info(f"bandits={bandits}")
    return plot_parameters


x = np.linspace(0, 1, 200)
plot_data = generate_data()

anni = FuncAnimation(
    fig=fig,
    func=animation,
    frames=NUM_ITERATIONS,
    init_func=init,
    interval=50,
)
anni.save(tmp_dir / "bandit_animation.mp4", writer="ffmpeg", fps=30)
plt.show()
