<p align="center">
  <img src="docs/ovon_task.jpg" width="700">
  <h1 align="center">HM3D-OVON: A Dataset and Benchmark for Open-Vocabulary Object Goal Navigation</h1>
  <h3 align="center">
    <a href="http://naoki.io/">Naoki Yokoyama</a>,
    <a href="https://ram81.github.io/">Ram Ramrakhya</a>,
    <a href="https://abhishekdas.com/">Abhishek Das</a>,
    <a href="https://faculty.cc.gatech.edu/~dbatra/">Dhruv Batra</a>,
    <a href="https://faculty.cc.gatech.edu/~sha9/">Sehoon Ha</a>
  </h3>
  <p align="center">
    <a href="http://naoki.io/portfolio/ovon.html">Project Website</a>
  </p>
</p>

## Overview

> **This repository is used to run [NaVILA](https://navila-bot.github.io/) on the [HM3D-OVON](https://naoki.io/portfolio/ovon.html) benchmark as a baseline.**

We present the Habitat-Matterport 3D Open Vocabulary Object Goal Navigation dataset (HM3D-OVON), a large-scale benchmark that broadens the scope and semantic range of prior Object Goal Navigation (ObjectNav) benchmarks. Leveraging the HM3DSem dataset, HM3D-OVON incorporates over 15k annotated instances of household objects across 379 distinct categories, derived from photo-realistic 3D scans of real-world environments.

In contrast to earlier ObjectNav datasets, which limit goal objects to a predefined set of 6–21 categories, HM3D-OVON facilitates the training and evaluation of models with an open set of goals defined through free-form language at test time. Through this open-vocabulary formulation, HM3D-OVON encourages progress towards learning visuo-semantic navigation behaviors capable of searching for any object specified by text.

We systematically evaluate and compare several different types of approaches on HM3D-OVON. We find that it can be used to train an open-vocabulary ObjectNav agent that achieves both higher performance and greater robustness to localization and actuation noise than the state-of-the-art ObjectNav approach. Videos available at: naoki.io/ovon.

---

## Table of Contents

1. [Installation](#installation)
2. [Downloading the Datasets](#downloading-the-datasets)
3. [Pre-trained Weights](#pre-trained-weights)
4. [Evaluation](#evaluation)
5. [Training](#training)
6. [NaVILA Evaluation](#navila-evaluation)
   - [Prerequisites](#prerequisites)
   - [Evaluation Commands](#evaluation-commands)
   - [Visualization](#visualization)
   - [Failure Analysis](#failure-analysis)
   - [Output Metrics](#output-metrics)
   - [Parallel Evaluation](#parallel-evaluation-6-gpus-load-balanced)
   - [How It Works](#how-it-works)

---

## Installation

Create the conda environment and install all dependencies. Mamba is recommended for faster installation:

```bash
conda_env_name=ovon
mamba create -n $conda_env_name python=3.7 cmake=3.14.0 -y
mamba install -n $conda_env_name \
  habitat-sim=0.2.3 headless pytorch=1.12.1 cudatoolkit=11.3 \
  -c pytorch -c nvidia -c conda-forge -c aihabitat -y

# Install this repo as a package
mamba activate $conda_env_name
pip install -e .

# Install frontier_exploration
cd frontier_exploration && pip install -e . && cd ..

# Install habitat-lab
git clone --branch v0.2.3 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines

pip install ftfy regex tqdm GPUtil trimesh seaborn timm scikit-learn einops transformers
pip install git+https://github.com/openai/CLIP.git
```

---

## Downloading the Datasets

First, set the following variables (no need to add to `.bashrc`):

```bash
MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
DATA_DIR=</path/to/ovon/data>
```

### Download HM3D 3D Scans

```bash
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR &&
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR
```

### Download OVON Navigation Episodes

The OVON navigation episodes can be found here: https://huggingface.co/datasets/nyokoyama/hm3d_ovon/

Decompress the `.tar.gz` file into `data/datasets/ovon/` so that the `hm3d` directory is located at `data/datasets/ovon/hm3d/`.

---

## Pre-trained Weights

The weights for the DagRL policy can be downloaded from the following link:

- `dagrl.pth`: https://drive.google.com/drive/folders/1U-tnPYQa81JbYHSlW1nyjiXOK8cE2Ki8?usp=sharing

---

## Evaluation

Run the following to evaluate:

```bash
python -m ovon.run \
  --run-type eval \
  --exp-config config/experiments/dagger_objectnav.yaml \
  habitat_baselines.eval_ckpt_path_dir=<path_to_ckpt>
```

---

## Training

Run the following to train:

```bash
python -m ovon.run \
  --run-type train \
  --exp-config config/experiments/dagger_objectnav.yaml
```

---

## NaVILA Evaluation

[NaVILA](https://navila-bot.github.io/) is a VLM-based navigation agent (LLaMA-3 8B + vision tower) that generates navigation actions from language and image input. The integration below evaluates NaVILA on the OVON ObjectNav benchmark without any RL training — the VLM drives action selection directly.

Key files:
- `ovon/trainers/navila_objectnav_trainer.py` — NaVILA trainer
- `config/experiments/navila_objectnav.yaml` — experiment config

### Prerequisites

- **Conda environment** with both OVON packages and NaVILA's `llava` package installed:
  ```bash
  PYTHON=/workspace/conda_envs/ovon-navila/bin/python
  ```
  All commands below use `$PYTHON`. Substitute your actual path if different.

- **NaVILA checkpoint** at `/workspace/NaVILA/checkpoints/navila-llama3-8b-8f/` (set via `habitat_baselines.eval_ckpt_path_dir` in the config or on the command line).

- **Dataset files** at `data/datasets/ovon/hm3d/`:

  | Split | Data path |
  |-------|-----------|
  | `val_unseen` *(default)* | `data/datasets/ovon/hm3d/val_unseen/val_unseen_hard.json.gz` |
  | `val_seen` | `data/datasets/ovon/hm3d/val_seen/val_seen.json.gz` |
  | `val_seen_synonyms` | `data/datasets/ovon/hm3d/val_seen_synonyms/val_unseen_easy.json.gz` |

### Evaluation Commands

All commands should be run from the repo root. The base command is:

```bash
$PYTHON -m ovon.run \
  --run-type eval \
  --exp-config config/experiments/navila_objectnav.yaml
```

**To switch splits**, append both overrides (only `val_unseen` is the default; the others require both flags):

```bash
# val_seen
  habitat_baselines.eval.split=val_seen \
  habitat.dataset.data_path=data/datasets/ovon/hm3d/val_seen/val_seen.json.gz

# val_seen_synonyms
  habitat_baselines.eval.split=val_seen_synonyms \
  habitat.dataset.data_path=data/datasets/ovon/hm3d/val_seen_synonyms/val_unseen_easy.json.gz
```

Add `habitat_baselines.test_episode_count=N` to any command to limit the number of episodes evaluated (e.g. `=1` for a quick sanity check).

### Visualization

Each frame rendered shows `[RGB camera view | top-down trajectory map]` with per-step metrics overlaid. Three visualization modes are available and can be combined freely:

#### Live preview (browser, auto-refreshes every 500ms)

```bash
NAVILA_LIVE_PREVIEW=live_preview \
$PYTHON -m ovon.run --run-type eval \
  --exp-config config/experiments/navila_objectnav.yaml \
  habitat_baselines.test_episode_count=1
```

Open **http://localhost:8765** in a browser. Use `NAVILA_LIVE_PORT=9000` to change the port.

- **VS Code Remote SSH:** VS Code auto-detects the port — open the **PORTS** tab and click *Open in Browser*.
- **Plain SSH:** `ssh -L 8765:localhost:8765 user@remote-host`

#### Record every episode to disk

```bash
$PYTHON -m ovon.run --run-type eval \
  --exp-config config/experiments/navila_objectnav.yaml \
  'habitat_baselines.eval.video_option=["disk"]' \
  habitat_baselines.video_dir=videos/navila_test
```

Videos are saved to `habitat_baselines.video_dir` (default: `video_dir/navila_objectnav/`).

#### Smart trajectory saving (recommended for large runs)

Instead of recording every episode, set `NAVILA_TRAJ_SAVE_DIR` to save a curated subset — stratified by object category and outcome — with no extra config required.

```bash
NAVILA_TRAJ_SAVE_DIR=trajectories/val_seen_synonyms \
$PYTHON -m ovon.run --run-type eval \
  --exp-config config/experiments/navila_objectnav.yaml \
  habitat_baselines.eval.split=val_seen_synonyms \
  habitat.dataset.data_path=data/datasets/ovon/hm3d/val_seen_synonyms/val_unseen_easy.json.gz
```

Each episode is tagged and saved only if the per-category quota is not yet full:

| Tag | Condition | Default quota |
|-----|-----------|---------------|
| `success` | Stopped within 1.0 m of goal | 1 per category |
| `close_fail` | Failed, final distance < 0.5 m | 4 per category |
| `far_fail` | Failed, final distance ≥ 0.5 m | 4 per category |

Video filenames encode category, outcome, episode ID, and final distance:
```
armchair__far_fail__ep42__dtg3.21.mp4
backpack__success__ep107__dtg0.18.mp4
```

| Env var | Default | Description |
|---------|---------|-------------|
| `NAVILA_TRAJ_SAVE_DIR` | *(unset)* | Output directory; feature disabled if not set |
| `NAVILA_TRAJ_MAX_FAIL_PER_CAT` | `4` | Max failure videos per category |
| `NAVILA_TRAJ_MAX_SUCCESS_PER_CAT` | `1` | Max success videos per category |

Storage estimate for `val_seen_synonyms` (36 categories):

| Quota | Max videos | Approx size |
|-------|------------|-------------|
| 4 fail + 1 success *(default)* | 180 | ~1.8 GB |
| 2 fail + 1 success | 108 | ~1.1 GB |
| 1 fail + 1 success | 72 | ~720 MB |

### Failure Analysis

#### Step 1 — collect per-episode metrics during the run

Set `OVON_EPISODES_JSON` to write a JSON file with one entry per episode (success, SPL, distance to goal, object category, scene):

```bash
OVON_EPISODES_JSON=results/val_seen_synonyms.json \
NAVILA_TRAJ_SAVE_DIR=trajectories/val_seen_synonyms \
$PYTHON -m ovon.run --run-type eval \
  --exp-config config/experiments/navila_objectnav.yaml \
  habitat_baselines.eval.split=val_seen_synonyms \
  habitat.dataset.data_path=data/datasets/ovon/hm3d/val_seen_synonyms/val_unseen_easy.json.gz
```

#### Step 2 — run the analysis script

Save as `analyze_results.py` and run `python analyze_results.py`:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("results/val_seen_synonyms.json") as f:
    data = json.load(f)

df = pd.DataFrame(data.values())

# Overall metrics
print("=== Overall ===")
print(df[["success", "spl", "soft_spl", "distance_to_goal"]].mean().round(4))

# Per-category success rate (sorted worst → best)
cat_stats = (
    df.groupby("target")[["success", "spl", "soft_spl", "distance_to_goal"]]
    .mean().round(4).sort_values("success")
)
print("\n=== Per-category ===")
print(cat_stats.to_string())
print("\n5 hardest:\n", cat_stats.head(5))
print("\n5 easiest:\n", cat_stats.tail(5))

# Outcome classification
df["outcome"] = "far_fail"
df.loc[df["success"] == 1, "outcome"] = "success"
df.loc[(df["success"] == 0) & (df["distance_to_goal"] < 0.5), "outcome"] = "close_fail"
print("\n=== Outcome breakdown ===")
print(df["outcome"].value_counts(normalize=True).round(3))

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cat_stats["success"].plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Success rate per category")
axes[0].axvline(df["success"].mean(), color="red", linestyle="--", label="mean")
axes[0].legend()

sns.histplot(data=df, x="distance_to_goal", hue="outcome", bins=40, ax=axes[1], element="step")
axes[1].set_title("Distance to goal at episode end")

counts = df["outcome"].value_counts()
axes[2].pie(counts, labels=counts.index, autopct="%.0f%%", colors=sns.color_palette("pastel"))
axes[2].set_title("Outcome breakdown")

plt.tight_layout()
plt.savefig("results/analysis.png", dpi=150)
print("Saved results/analysis.png")
```

The script produces:
- **Terminal:** overall averages, per-category table sorted by success rate, 5 hardest/easiest categories, outcome breakdown percentages
- **`results/analysis.png`:** per-category success bar chart, distance-to-goal histogram by outcome, outcome pie chart

The `close_fail` vs `far_fail` split is the key diagnostic: if `far_fail` dominates (>80%), the problem is exploration — the agent never finds the object. If `close_fail` is large (>15%), the problem is the STOP decision — the agent reaches the goal but doesn't stop.

### Output Metrics

Metrics are displayed in two contexts:

- **Live preview / per-step overlay** — values update every step and reflect the *current state within the episode* (not final results)
- **Terminal / episode end** — printed once per episode after STOP is called, then aggregated across all episodes at the end of the run

| Metric | Per-step meaning | Episode-end meaning |
|--------|-----------------|---------------------|
| `distance_to_goal` | Current geodesic distance (metres) to the nearest goal viewpoint right now | Final geodesic distance at the step the agent stopped |
| `success` | 0 while the episode is running; becomes 1 only after a successful STOP | 1 if agent stopped within 1.0 m of any valid goal viewpoint, 0 otherwise |
| `spl` | Always 0 while running — SPL is computed only on a successful stop | Success weighted by Path Length: `success × optimal_path / max(actual_path, optimal_path)` |
| `soft_spl` | Path efficiency so far; 0 if agent hasn't made net progress toward goal | Path efficiency regardless of success — always ≥ `spl` |
| `distance_to_goal_reward` | `previous_distance − current_distance` this step; negative means no progress | Cumulative reward signal for the episode |
| `top_down_map.agent_angle` | Agent heading in radians at this step (`π ≈ 3.14` = facing backward) | Agent's final heading at stop time |

**Reference from the OVON paper:** `success` > 0.20 and `spl` > 0.10 on `val_unseen` is competitive.

### Parallel Evaluation (6 GPUs, load-balanced)

For the full `val_unseen` split (~3000 episodes), use the load-balanced parallel eval script which distributes scenes optimally across 6 GPUs (~33 hrs total):

```bash
bash scripts/eval/eval_val_unseen_balanced.sh
```

Results are written to `data/eval/val_unseen/<timestamp>/` and include:
- `episode_metrics.json` — merged metrics for all episodes
- `failure_analysis.html` — Sankey failure analysis chart
- `videos/` — curated trajectory videos (global quota: 5 fail + 2 success per category)

**Before your first full run, validate the setup with the smoke test** (~20 min, 3 episodes per GPU):

```bash
bash scripts/eval/smoke_test_balanced.sh
```

The smoke test verifies no crashes occur at end-of-run, that `episode_metrics.json` is written correctly per worker, and that the merge and Sankey generation work. All 20 checks must pass before committing to the full run.

**Monitor progress** while the eval is running:

```bash
bash scripts/eval/watch_progress.sh
```

This refreshes every 60 seconds and shows a progress bar, done/total episode count, elapsed time, and ETA for each GPU worker.

**Recommended tmux workflow:**

```bash
# Window 1 — run the eval
tmux new-session -s eval
bash scripts/eval/eval_val_unseen_balanced.sh

# Window 2 — monitor progress (Ctrl+B C to open new window)
bash scripts/eval/watch_progress.sh

# Detach and come back later
# Ctrl+B D        →  detach
# tmux attach -t eval  →  reattach
```

### How It Works

The NaVILA integration replaces the RL policy's `act()` call with a VLM inference step:

1. The current RGB frame is appended to a rolling history of past frames
2. The history is sampled/padded to 8 frames and passed to NaVILA alongside the prompt: *"Navigate to and find a `<object>`. When you are close to the `<object>`, stop."*
3. NaVILA outputs one of the canonical text actions below
4. The direction and degree are parsed; turns larger than 15° are broken into multiple 15°-step actions queued before re-querying the model; forward moves always trigger a fresh query
5. Episode object categories are preloaded from `content/*.json.gz` at startup since `current_episodes()` returns stripped objects without category info

**Canonical outputs:**

| Raw text output | Action | Steps executed |
|---|---|---|
| `The next action is stop.` | STOP | — |
| `The next action is move forward 25 cm.` | FORWARD | 1 × 25 cm |
| `The next action is move forward 50 cm.` | FORWARD | 1 × 25 cm (re-queried after each) |
| `The next action is move forward 75 cm.` | FORWARD | 1 × 25 cm (re-queried after each) |
| `The next action is turn left 15 degree.` | TURN_LEFT | 1 × 15° |
| `The next action is turn left 30 degree.` | TURN_LEFT | 2 × 15° (queued) |
| `The next action is turn left 45 degree.` | TURN_LEFT | 3 × 15° (queued) |
| `The next action is turn right 15 degree.` | TURN_RIGHT | 1 × 15° |
| `The next action is turn right 30 degree.` | TURN_RIGHT | 2 × 15° (queued) |
| `The next action is turn right 45 degree.` | TURN_RIGHT | 3 × 15° (queued) |

**Parsing notes:**
- Action keywords matched case-insensitively: `stop` / `is move forward` / `is turn left` / `is turn right`
- Turn degrees extracted with `r"turn\s+(?:left|right)\s+(\d+(?:\.\d+)?)\s*degree"`, converted to `n_steps = max(1, round(degree / 15))`
- Forward moves always execute one step at a time with a fresh VLM query
- Unrecognised output defaults to `FORWARD`
