"""Load bundled examples or supervised tabular datasets from OpenML."""

from importlib.resources import files
from numbers import Integral

import pandas as pd
from sklearn.datasets import fetch_openml


_BUNDLED_DATASETS = {
    "iris": "datasets/iris.csv",
    "titanic": "datasets/titanic.csv",
    "auto-mpg": "datasets/auto-mpg.csv",
    "artificial_tree": "datasets/artificial_tree.csv",
}

# Pin the OpenML dataset ID and target so a named example stays reproducible.
_OPENML_DATASETS = {
    "credit-g": (31, "class"),
    "wine-quality-red": (40691, "class"),
}


def load_data(name="iris", *, target=None):
    """Return a dataset as a DataFrame with its target in the first column.

    ``name`` may be a bundled example name, a named OpenML example, or a
    positive OpenML dataset ID. Named examples have fixed targets. For an ID,
    ``target`` selects a column; otherwise the OpenML default target is used.
    OpenML datasets are downloaded on first use and cached by scikit-learn.
    """
    if isinstance(name, str):
        if target is not None:
            raise ValueError("target can only be specified for an OpenML dataset ID")
        if name in _BUNDLED_DATASETS:
            return pd.read_csv(files("rfphate").joinpath(_BUNDLED_DATASETS[name]))
        if name in _OPENML_DATASETS:
            data_id, target_column = _OPENML_DATASETS[name]
        else:
            known_names = ", ".join(repr(item) for item in (*_BUNDLED_DATASETS, *_OPENML_DATASETS))
            raise NameError(f"{name!r} is not recognized. Known names are {known_names}.")
    elif isinstance(name, Integral) and not isinstance(name, bool) and name > 0:
        data_id = int(name)
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ValueError("target must be a nonempty column name")
        target_column = target if target is not None else "default-target"
    else:
        raise ValueError("name must be a known dataset name or a positive OpenML dataset ID")

    dataset = fetch_openml(
        data_id=data_id, target_column=target_column, as_frame=True, cache=True
    )
    features = dataset.data
    labels = dataset.target

    if not isinstance(features, pd.DataFrame):
        raise ValueError(f"OpenML dataset {data_id} did not provide tabular features")
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            raise ValueError(
                f"OpenML dataset {data_id} has multiple targets; pass target='column_name'"
            )
        labels = labels.iloc[:, 0]
    if labels is None:
        raise ValueError(
            f"OpenML dataset {data_id} has no default target; pass target='column_name'"
        )
    if not isinstance(labels, pd.Series):
        raise ValueError(f"OpenML dataset {data_id} did not provide a tabular target")

    target_name = (
        target_column if target_column != "default-target"
        else labels.name if labels.name is not None else "target"
    )
    return pd.concat([labels.rename(target_name), features], axis=1)
