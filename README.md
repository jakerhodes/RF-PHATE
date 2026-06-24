# RF-PHATE

RF-PHATE creates supervised, low-dimensional embeddings from forest proximity
matrices and a PageRank PHATE operator. It is designed for exploratory data
analysis when the target variable should guide the geometry of the embedding.

The current implementation builds proximities with
[`forestgeom`](https://github.com/JakeSRhodesLab/forestgeom) `ForestProximity`
objects, using scikit-learn forest estimators under the hood.

A repository with scripts to replicate the main quantification comparisons from
the paper is available at
[RF-PHATE-Quantification](https://github.com/jakerhodes/RF-PHATE-Quantification).

## Installation

Install RF-PHATE from GitHub with `pip`:

```bash
pip install git+https://github.com/jakerhodes/RF-PHATE
```

Core dependencies include `forestgeom`, `graphtools`, `phate`,
`scikit-learn`, `numpy`, `scipy`, and `pandas`. Demo plotting additionally uses
`seaborn`, `plotly`, and `nbformat`.

## Forestgeom Refactor

RF-PHATE now delegates random forest proximity construction to
`forestgeom.ForestProximity`. The `RFPHATE` class builds a scikit-learn ensemble,
wraps it in `ForestProximity`, and passes the resulting proximity matrix to
`PageRankPHATE`.

Important API points:

- `prediction_type` selects the estimator family. Use `"classification"` or
  `"regression"`.
- `model_type` selects the base ensemble: `"rf"` for random forests, `"et"` for
  extra trees, and `"gbt"` for gradient boosted trees.
- `kernel_method` is passed to the `forestgeom` proximity weighting scheme.
  The default is `"gap"`; use `"uniform"` for unweighted forest proximity.
- `random_state`, `verbose`, and `n_jobs` are explicit RF-PHATE parameters and
  are passed through to both the forest estimator and `PageRankPHATE` where
  supported. scikit-learn gradient boosting estimators do not support `n_jobs`.
- `forest_params` are passed to the underlying scikit-learn ensemble.
- `proximity_params` are passed to `forestgeom.ForestProximity`.
- `phate_params` are passed to `PageRankPHATE`.
- `transform(data)` embeds new observations using the fitted forest proximity
  model and PHATE graph interpolation.

## Quick Demo

```python
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import rfphate

Path("figures").mkdir(exist_ok=True)

data = rfphate.load_data("titanic")
x, y = rfphate.dataprep(data)

rfphate_op = rfphate.RFPHATE(
    random_state=42,
    n_jobs=-1,
    verbose=1,
    kernel_method="gap",
    model_type="rf",
    force_symmetric=True,
    adjust_diagonal=True,
    forest_params={
        "n_estimators": 100,
    },
    phate_params={
        "mds_solver": "sgd",
    },
)

emb = rfphate_op.fit_transform(x, y)

plot_data = data.assign(
    **{
        "RF-PHATE 1": emb[:, 0],
        "RF-PHATE 2": emb[:, 1],
        "Pclass": data["Pclass"].astype(str),
    }
)
```

Color the embedding by passenger class and use marker style for survival. The
saved PNG uses the same `plot_data` coordinates built from `emb` above:

```python
plt.figure(figsize=(7, 5.5), dpi=150)
plot = sns.scatterplot(
    data=plot_data,
    x="RF-PHATE 1",
    y="RF-PHATE 2",
    hue="Pclass",
    style="Survived",
    markers={"survived": ".", "died": "X"},
    alpha=0.8,
    palette="Dark2",
)
plot.set_title("RF-PHATE Embedding Colored by Pclass")
plot.set_xlabel("RF-PHATE 1")
plot.set_ylabel("RF-PHATE 2")
plt.savefig("figures/titanic_pclass.png", bbox_inches="tight")
plt.show()
```

![Titanic passenger class embedding](figures/titanic_pclass.png)

Color the same embedding by passenger sex. The saved PNG uses the same
`plot_data` coordinates built from `emb` above:

```python
plt.figure(figsize=(7, 5.5), dpi=150)
plot = sns.scatterplot(
    data=plot_data,
    x="RF-PHATE 1",
    y="RF-PHATE 2",
    hue="Sex",
    style="Survived",
    markers={"survived": ".", "died": "X"},
    alpha=0.9,
    palette="Dark2",
)
plot.set_title("RF-PHATE Embedding Colored by Sex")
plot.set_xlabel("RF-PHATE 1")
plot.set_ylabel("RF-PHATE 2")
plt.savefig("figures/titanic_sex.png", bbox_inches="tight")
plt.show()
```

![Titanic sex embedding](figures/titanic_sex.png)

## Train/Test Embedding

Fit RF-PHATE on training data, then embed held-out observations with
`transform`:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

rfphate_split_op = rfphate.RFPHATE(
    random_state=42,
    n_jobs=-1,
    verbose=1,
    kernel_method="gap",
    model_type="rf",
    force_symmetric=True,
    adjust_diagonal=True,
)

emb_train = rfphate_split_op.fit_transform(x_train, y_train)
emb_test = rfphate_split_op.transform(x_test)
```

## Citation

If you find RF-PHATE useful, please cite:

Rhodes, J.S., Aumon, A., Morin, S., et al. Gaining Biological Insights through
Supervised Data Visualization. bioRxiv (2023).
https://doi.org/10.1101/2023.11.22.568384.
