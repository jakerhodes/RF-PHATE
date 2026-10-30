import numpy as np
import pandas as pd
import pytest

from rfphate import dataprep, load_data


@pytest.mark.parametrize("name,shape", [
    ("iris", (150, 5)), ("titanic", (712, 8)),
    ("auto-mpg", (392, 7)), ("artificial_tree", (1440, 61)),
])
def test_packaged_datasets(name, shape):
    data = load_data(name)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == shape
    x, y = dataprep(data)
    assert x.shape == (shape[0], shape[1] - 1)
    assert len(y) == shape[0]
    assert np.isfinite(x).all()


def test_default_and_invalid_dataset_names():
    pd.testing.assert_frame_equal(load_data(), load_data("iris"))
    with pytest.raises(NameError, match="not recognized"):
        load_data("missing")


@pytest.mark.parametrize("transform,expected", [
    ("normalize", [[0, 0, 7], [0.5, 1, 7], [1, 0, 7]]),
    ("standardize", [[-1, -1 / np.sqrt(3), 7], [0, 2 / np.sqrt(3), 7],
                     [1, -1 / np.sqrt(3), 7]]),
])
def test_preprocessing_values_and_input_preservation(transform, expected):
    data = pd.DataFrame({
        "label": ["yes", "no", "yes"], "numeric": [10, 20, 30],
        "category": ["a", "b", "a"], "constant": [7, 7, 7],
    })
    original = data.copy(deep=True)
    x, y = dataprep(data, transform=transform)
    np.testing.assert_allclose(x, expected)
    np.testing.assert_array_equal(y, [1, 0, 1])
    pd.testing.assert_frame_equal(data, original)


def test_label_selection_and_unlabelled_features():
    data = pd.DataFrame({"feature": [5, 10, 15], "label": [4, 5, 6]})
    x, y = dataprep(data, label_col_idx=1)
    np.testing.assert_allclose(x.ravel(), [0, 0.5, 1])
    np.testing.assert_array_equal(y, [4, 5, 6])
    x = dataprep(data, label_col_idx=None)
    np.testing.assert_allclose(x, [[0, 0], [0.5, 0.5], [1, 1]])
