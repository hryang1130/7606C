"""Two-level bootstrap of the success difference between two data sizes.

Resamples training seeds, then the 100 evaluation episodes of each resampled
run (as a binomial draw from the run's success rate, since only per-run rates
are recorded here), and reports the mean difference with its 95% interval.
Episodes are not paired between the two sizes.

The PickCube lines reproduce the intervals of report_pickcube_yhr.md
(+26.0 [19.8, 32.2] and +8.2 [1.4, 15.0]); the PlaceSphere rates are from
dp-manip/docs/experiment-progress.zh-CN.md sections 6 and 15.
Usage: python3 scripts/bootstrap_adjacent.py
"""

import numpy as np

EPISODES, SAMPLES = 100, 10000

RATES = {
    "PickCube": {
        100: [0.17, 0.18, 0.19, 0.21, 0.21],
        200: [0.50, 0.43, 0.46, 0.42, 0.45],
        400: [0.53, 0.52, 0.51, 0.55, 0.56],
    },
    "PlaceSphere": {
        25: [0.08, 0.01, 0.05],
        50: [0.30, 0.20, 0.26],
        100: [0.32, 0.41, 0.36, 0.39, 0.32],
        200: [0.79, 0.84, 0.77, 0.68, 0.75],
        400: [0.96, 0.97, 0.97, 0.95, 0.96],
    },
}


def difference(low, high, rng):
    low, high = np.asarray(low), np.asarray(high)
    draws = np.empty(SAMPLES)
    for i in range(SAMPLES):
        a = low[rng.integers(0, len(low), len(low))]
        b = high[rng.integers(0, len(high), len(high))]
        draws[i] = rng.binomial(EPISODES, b).mean() - rng.binomial(EPISODES, a).mean()
    return (high.mean() - low.mean()) * 100, np.percentile(draws, [2.5, 97.5])


def main() -> None:
    rng = np.random.default_rng(0)
    for task, rates in RATES.items():
        sizes = sorted(rates)
        for low, high in zip(sizes, sizes[1:]):
            point, (lo, hi) = difference(rates[low], rates[high], rng)
            print(f"{task} {low}->{high}: {point:+.1f} [{lo:+.1f}, {hi:+.1f}]")


if __name__ == "__main__":
    main()
