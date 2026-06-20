"""Build train/val/test splits for the ARC-AGI-2 CodeEvolver run.

Split policy (deliberately different from the ARC-AGI-1 pool-and-reshuffle):
the 120 public *evaluation* tasks are the precious, intended held-out set, so
they become the TEST set untouched and are never used as optimization feedback.
The 1000 public *training* tasks are split into train (prototype) and val
(the optimization feedback signal the loop is allowed to fit), seed-fixed.
"""

import os
import json
import random
import glob

random.seed(42)

VAL_SIZE = 200  # of the 1000 training tasks; remaining 800 -> train


def load_tasks(dir_path):
    tasks = []
    for filepath in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        with open(filepath, "r") as f:
            data = json.load(f)
        data["task_id"] = os.path.basename(filepath).split(".")[0]
        tasks.append(data)
    return tasks


training_tasks = load_tasks("data/training")     # 1000
evaluation_tasks = load_tasks("data/evaluation")  # 120

random.shuffle(training_tasks)
val_data = training_tasks[:VAL_SIZE]
train_data = training_tasks[VAL_SIZE:]
test_data = evaluation_tasks  # held-out, untouched

os.makedirs("data_splits", exist_ok=True)
os.makedirs("tests/testdata", exist_ok=True)

with open("data_splits/trainset.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("data_splits/valset.json", "w") as f:
    json.dump(val_data, f, indent=2)

# Path-guarded held-out test file (the 120 public-eval tasks).
with open("tests/testdata/testset.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Split completed: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
