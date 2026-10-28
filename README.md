# Digit Recognizer with PyTorch – Assignment 2

**Author:** Manvi Gupta
**Kaggle username:** manvig1
**Kaggle public score:** 0.99471

Handwritten digit classification (0–9) for the Kaggle
[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition.
The project compares an MLP baseline, a CNN, and an improved CNN, all built with PyTorch.
The full report is in [`memory.pdf`](memory.pdf).

## Results

| Model | Best validation accuracy |
|---|---|
| MLP (baseline) | 0.97xx |
| CNN (2 conv blocks) | 0.9900 |
| CNN + BatchNorm + Dropout | 0.99xx |
| CNN + BatchNorm + Dropout + augmentation + OneCycleLR | **0.9952** |

The final model scored **0.99471** on the Kaggle public leaderboard.

## Project structure

```
├── README.md
├── memory.pdf  # report
├── pyproject.toml  # package metadata and dependencies
├── requirements.txt
├── data/
│   └── README.md  # how to download the Kaggle data
├── notebooks/
│   └── main.ipynb  # main notebook (executed, imports the package)
├── results/
│   ├── submission.csv  # file submitted to Kaggle
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── misclassified.png
├── src/digits/
│   ├── __init__.py
│   ├── data.py # Dataset, normalisation, stratified split, augmentation
│   ├── models.py # MLP and CNN (optional BatchNorm + Dropout)
│   ├── train.py # training loop, evaluation, fit()
│   ├── predict.py  # predictions and submission file
│   └── plots.py  # learning curves, confusion matrix, misclassified digits
└── tests/
    └── test_models.py  # checks every model returns logits of shape [B, 10]
```

## Installation

Requires Python 3.10 or newer. From the project root:

```bash
# 1. Create and activate an environment (conda or venv)
conda create -n digits python=3.11 -y
conda activate digits

# 2. Install the package and its dependencies (editable mode)
pip install -e ".[dev]"
```

Alternatively: `pip install -r requirements.txt` followed by `pip install -e .`

## Data

The CSV files are not included. See [`data/README.md`](data/README.md), or with the Kaggle CLI:

```bash
kaggle competitions download -c digit-recognizer -p data/
unzip data/digit-recognizer.zip -d data/
```

`data/` should then contain `train.csv` and `test.csv`.

## How to run

1. Open `notebooks/main.ipynb` with the environment above as the kernel.
2. Run all cells. The notebook loads the data, trains the four models, plots the analysis
   figures and writes `results/submission.csv`.

Training uses a GPU automatically if available (CUDA or Apple MPS), otherwise the CPU.
On an Apple M-series laptop, the full notebook takes about 10–15 minutes.

## Tests

```bash
pytest
```

## Reproducibility

All randomness (data split, shuffling, weight initialisation) is seeded with `42`.
On Apple MPS some GPU operations are not fully deterministic, so results can differ
slightly in the last decimals between runs.