
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

> **This repository is a fork of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm).**
> It extends the upstream code with:
> - **Comprehensive onboarding documentation** — a step-by-step reproduction guide covering environment setup, dependency quirks, training, and evaluation (see [`results/REPORT.md`](results/REPORT.md))
> - **TwoRoom experiment results** — trained checkpoint, training metrics, loss curves, and 50 rollout videos from a complete end-to-end run on a single RTX 4090 (see [TwoRoom Results](#tworoom-results) below)

### Training loss dashboard

![Training loss dashboard](https://raw.githubusercontent.com/az9713/le-wm-tworoom/main/results/loss_dashboard.png)

The 2×2 panel shows all loss components over the full 8-epoch run (41,103 steps). Dashed vertical lines mark epoch boundaries.

- **Top-left — total loss** (`pred_loss + 0.09 × sigreg_loss`): drops 14× in epoch 0, then descends smoothly to ~0.17. Red dots are per-epoch validation loss; they track the training curve closely after a transient spike at epoch 1.
- **Top-right — prediction loss** (`pred_loss`): the predictor's MSE on the next latent embedding. Monotone descent from 0.45 → 0.008 with no collapse signature (a collapsing encoder would drive this to ~0 immediately while `sigreg` stayed high).
- **Bottom-left — SIGReg loss (linear)**: Gaussianity regularizer on the encoder output. Dominant early (31.9 at step 0), falls rapidly. The linear axis compresses all post-epoch-1 structure — use the log panel.
- **Bottom-right — SIGReg loss (log)**: same data on a log y-axis. The epoch-1 validation spike to 71.3 is clearly visible; by epoch 2 the held-out curve rejoins the training trace and both descend together. This single-bump-then-stabilize pattern is the regime change LeWM is specifically designed to survive without collapse.

### Sample rollout videos

Each video is a 3.3-second, 15-fps clip showing a 2×2 grid: **top-left** = agent under LeWM+CEM control, **top-right** = goal image, **bottom-left** = expert demo from the same start, **bottom-right** = goal image repeated.

**Episode 0 — clean one-room-to-one-room crossing**

https://github.com/user-attachments/assets/a3af7a1b-6596-48bf-a08c-e1fec538c630

The agent (red dot, top-left) starts in the left room and needs to reach a goal position in the right room. It navigates directly through the doorway within the first ~20 steps, matching the expert trajectory (bottom-left) closely. CEM in the 192-dim latent correctly plans a straight-line path when the doorway is well-aligned with the start/goal axis.

**Episode 10 — crossing with an off-axis approach**

https://github.com/user-attachments/assets/df677f61-a7ff-4e3c-adba-ba26438fdb4d

The start and goal positions are not colinear with the doorway, requiring the agent to first reposition laterally before committing to the cross-room transition. The world model plans the two-phase manoeuvre (approach angle correction → doorway traversal) within the 5-step receding horizon. Compare with the expert (bottom-left): the CEM path is not identical but reaches the same goal within the 50-step budget.

---

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## TwoRoom Results

A complete end-to-end reproduction of the TwoRoom experiment was run on a single **RTX 4090 (24 GB)** using the upstream code with no model changes.

| Run | Hardware | Epochs | TwoRoom success rate | Wall clock |
|:---:|:---:|:---:|:---:|:---:|
| Paper | L40S 48 GB | — | **97 %** | "few hours" |
| [Tonbi (community)](https://www.youtube.com/watch?v=VQ15-MhZE2k) | RTX 3060 12 GB | 4 | 92 % | ~8 h |
| **This fork** | **RTX 4090 24 GB** | **8** | **94 %** | **~2 h 45 min** |

**47 of 50 episodes succeeded** (seed 42, `CEMSolver`, horizon=5, 50-step budget per episode).

### Results artifacts (`results/`)

| File | Description |
|---|---|
| [`REPORT.md`](results/REPORT.md) | Full run report — hardware audit, setup recipe, training loss analysis, evaluation discussion, failure case breakdown, cost breakdown, lessons learned |
| `tworoom_results.txt` | Raw eval output: `success_rate: 94.0`, per-episode boolean array, eval config |
| `training_metrics.csv` | Lightning CSV — every 50 training steps and every epoch boundary for all loss components |
| `training_config.yaml` | Frozen Hydra training config snapshot |
| `train.log` / `eval.log` | Full stdout/stderr from training and evaluation |
| `lewm_epoch_8_object.ckpt` | Trained model checkpoint (epoch 8, loadable via `stable_worldmodel`) |
| `rollout_0.mp4` … `rollout_49.mp4` | Per-episode rollout videos (448×448, 15 fps, 2×2 grid: agent / expert / goal) |
| `loss_dashboard.png` | 2×2 training loss panel with epoch-boundary overlays |
| `validation_bars.png` | Per-epoch validation loss comparison |

### Loading the checkpoint

```python
import stable_worldmodel as swm

cost = swm.policy.AutoCostModel(
    'lewm_epoch_8',
    cache_dir='results'   # path to the results/ directory
)
```

---

## Onboarding Documentation

[`results/REPORT.md`](results/REPORT.md) is a detailed reproduction guide that covers:

The YouTube video [**"I Reproduced LeCun's JEPA World Model That Doesn't Predict Tokens"**](https://www.youtube.com/watch?v=VQ15-MhZE2k) by Tonbi is a helpful companion — it walks through a similar end-to-end setup on an RTX 3060 and provides useful commentary on the training dynamics.

1. **Hardware and software audit** — exact versions of Python, PyTorch, CUDA, and torchvision validated for this run
2. **Clean setup recipe** — copy-pasteable command sequence from a fresh Ubuntu 22.04 pod to a running evaluation
3. **Dependency issues and fixes** — eight issues encountered with root-cause analysis and exact resolutions, including:
   - `stable_pretraining` torchvision v2 API incompatibility (sed patch)
   - `gymnasium[all]` / `gym==0.21.0` setuptools breakage (workaround)
   - RunPod hidden workspace quota with `/dev/shm` mitigation
   - Lightning cross-device checkpoint failure (Hydra override)
4. **Training loss analysis** — per-epoch trajectory interpretation, validation spike explanation, collapse-absence verification
5. **Evaluation walk-through** — CEM config, per-episode outcome, failure case analysis, rollout video guide
6. **Cost breakdown** — $0.69/hr RTX 4090, ~$3.45 total for a full 5-hour session

---

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
