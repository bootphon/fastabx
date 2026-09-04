"""Reproduce a `fastabx` CLI run with the Python API, then redo it with a bootstrap.

Equivalent CLI call:

    fastabx /scratch2/mpoli/assets/item/triphone-dev-clean.item ./features/11/ --max-size-group 10

which resolves to the CLI defaults: speaker="within", context="within", distance="angular",
frequency=50, seed=0.
"""

import numpy as np

from fastabx import Bootstrap, Dataset, Score, Task

ITEM = "/scratch2/mpoli/assets/item/triphone-dev-clean.item"
FEATURES = "./features/11/"
FREQUENCY = 50
SEED = 0
DISTANCE = "angular"
N_REPLICATES = 1000

BY = ["prev-phone", "next-phone", "speaker"]
LEVELS = [("next-phone", "prev-phone"), "speaker"]

dataset = Dataset.from_item(ITEM, FEATURES, FREQUENCY)
task = Task(
    dataset,
    on="#phone",
    by=BY,
    across=None,
)
score = Score(task, DISTANCE)
bootstrapped = Score(task, DISTANCE, bootstrap=Bootstrap(N_REPLICATES, seed=SEED))
point = bootstrapped.collapse(levels=LEVELS)
replicates = bootstrapped.bootstrap_collapse(levels=LEVELS)
low, high = bootstrapped.confidence_interval(levels=LEVELS)

print("=== standard ===")
print(f"ABX error rate: {score.collapse(levels=LEVELS):.3%}")
print(f"\n=== same, with bootstrap ({N_REPLICATES} replicates) ===")
print(f"ABX error rate: {point:.3%}")
print(f"bootstrap SD:   {np.nanstd(replicates, ddof=1):.3%}")
print(f"95% CI:         [{low:.3%}, {high:.3%}]")
