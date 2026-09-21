# Influmatch

Research code for **Influmatch**, an influence-guided relational consistency
method for semi-supervised medical image classification. The project extends
[USB: A Unified Semi-Supervised Learning Benchmark](https://github.com/microsoft/Semi-supervised-learning)
with a closed-form, last-layer influence estimator and adapters for several
SSL algorithms.

> This is a research repository, not the official USB distribution. The
> original USB documentation and benchmark claims do not describe this fork.

## Overview

Influmatch augments a standard SSL objective with a relation-consistency term.
For each unlabeled query, it uses labeled candidates from the current batch to
form a compact set of proponents and opponents. Candidate relations are scored
by combining representation similarity with a detached closed-form influence
signal. The strong-view representation is then trained to match the
weak-view relation distribution.

The main implementation follows these principles:

- **Closed-form influence approximation:** uses the classifier-head input
  features and cross-entropy logit gradients; no Hessian inversion or stored
  training trajectory is required.
- **Detached influence guidance:** influence scores select and weight
  relations, while the differentiable cosine relation on the strong view
  provides the gradient path.
- **Paired comparison:** the same base SSL algorithm, data split, augmentation,
  optimizer, and training budget are used with and without the Influmatch term.

The primary method is `fixmatch_ifcf`. Implementations are also provided for
AdaMatch, FlexMatch, FreeMatch, ReFixMatch, SoftMatch, and selected exploratory
variants.

## Repository Layout

```text
semilearn/algorithms/*/*_ifcf.py   Influmatch adapters for SSL algorithms
semilearn/algorithms/utils/closed_form_ifrank.py
                                   Shared closed-form IF and relation loss
train_ifcf.py                     Launcher that registers Influmatch methods
config/usb_cv/                    Dataset and algorithm configurations
run_*.sh                          Reproduction and experiment queue scripts
tools/                            Evaluation, audit, and plotting utilities
results/                          Local experiment outputs (gitignored)
saved_models/                     Local checkpoints (gitignored)
```

`SIMMATCH_IF_V2_*` files document exploratory SimMatch extensions. They are
kept separate from the Influmatch V1 main line and should not be interpreted as
the final method.

## Environment

The experiments were developed with Python 3.8, PyTorch, and CUDA on Linux.

```bash
git clone git@github.com:xxzcs/SSL-IF.git
cd SSL-IF

conda create -n influmatch python=3.8 -y
conda activate influmatch
pip install -r requirements.txt
pip install -e .
```

Install a PyTorch build compatible with the local CUDA driver before running
training. The exact CUDA/PyTorch combination is environment dependent.

## Data Preparation

The BUS, GDPH, and TN5000 image data and split files are **not distributed**
with this repository. They may be subject to dataset-specific access or usage
restrictions.

Create dataset-specific configuration files under `config/` and set:

```yaml
data_dir: /path/to/datasets
dataset: bus                  # or gdph / tn5000
lpath: /path/to/labeled_split.pth
ulpath: /path/to/unlabeled_split.pth
num_classes: 2
```

Keep every sample and its augmented copies within the same labeled/unlabeled
partition. Do not use test data for split construction, checkpoint selection,
or hyperparameter tuning.

## Training

### Baseline

Run a baseline configuration through the USB launcher:

```bash
python train.py --c path/to/fixmatch_config.yaml
```

### Influmatch

Set `algorithm: fixmatch_ifcf` in a copy of the corresponding FixMatch YAML,
then use the Influmatch launcher:

```bash
python train_ifcf.py --c path/to/fixmatch_ifcf_config.yaml
```

The essential Influmatch options are:

```yaml
algorithm: fixmatch_ifcf
ifrank_loss_weight: 1.0
corrT: 0.9
num_references: 4
ref_select: by_instance
ref_cand_k: 8
ifrank_combine: multiply_balanced
if_lambda: 1.0
csim_lambda: 1.0
if_tracin_scale: 1.0
ifrank_score_mode: fused
if_target: soft
ifrank_warmup_epochs: 1
ifrank_warmup_mode: zero
```

`if_tracin_scale: 0.0` explicitly disables the influence signal and should be
used only for a cosine-only control. `ifrank_score_mode: if` is intentionally
unsupported in the closed-form implementation because detached IF-only scores
do not provide a valid gradient path to the representation model.

Dataset-specific launch scripts are available as `run_bus_*.sh`,
`run_gdph_*.sh`, and `run_tn5000_*.sh`. They encode internal file paths and
experiment queues; review and adapt them before use on another machine.

## Evaluation and Reporting

Checkpoints and raw predictions are not versioned. Run evaluation only against
the held-out test set after fixing the training protocol. For paired method
comparisons, report the same checkpoint rule and thresholding rule for the
baseline and Influmatch variant.

The project uses five primary metrics for binary classification:

- AUC
- Accuracy
- Sensitivity
- Specificity
- F1 score

Use the scripts in `tools/` and the dataset-specific evaluation helpers rather
than mixing summaries generated under different split, checkpoint, or threshold
protocols.

## Reproducibility Notes

- Fix random seeds and record the labeled/unlabeled split seed separately from
  the training seed.
- Store generated YAML files, command lines, and evaluation settings with each
  experiment.
- Keep `latest` versus validation-selected checkpoint reporting consistent
  within a table.
- Generated data, checkpoints, TensorBoard logs, CSV summaries, and figures
  are intentionally excluded from Git unless explicitly curated for release.

## Acknowledgments

This repository is built on and substantially reuses the code structure of
[USB](https://github.com/microsoft/Semi-supervised-learning). Please cite USB
when using the shared benchmark infrastructure or baseline implementations:

```bibtex
@inproceedings{wang2022usb,
  title     = {USB: A Unified Semi-supervised Learning Benchmark for Classification},
  author    = {Wang, Yidong and others},
  booktitle = {NeurIPS},
  year      = {2022}
}
```

Please also follow the licenses and data-use terms of USB and each dataset used
in an experiment.
