"""OpenML loader contracts; routine tests never contact OpenML."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rfphate import dataprep, load_data


@pytest.fixture
def fake_openml(monkeypatch):
    calls = []
    features = pd.DataFrame({
        "age": [20, 40],
        "status": pd.Categorical(["single", "married"]),
    })
    labels = pd.Series(pd.Categorical(["good", "bad"]), name="class")

    def fetch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=features, target=labels)

    monkeypatch.setattr(import_module("rfphate.load_data"), "fetch_openml", fetch)
    return calls, features, labels


@pytest.mark.parametrize("name,data_id", [
    ("credit-g", 31), ("wine-quality-red", 40691),
])
def test_named_openml_examples_use_pinned_id_and_target(fake_openml, name, data_id):
    calls, features, labels = fake_openml
    result = load_data(name)
    assert calls == [dict(
        data_id=data_id, target_column="class", as_frame=True, cache=True
    )]
    pd.testing.assert_frame_equal(result, pd.concat([labels, features], axis=1))
    assert result.columns[0] == "class"
    assert isinstance(result["status"].dtype, pd.CategoricalDtype)
    x, y = dataprep(result)
    assert x.shape == (2, 2) and np.isfinite(x).all()
    np.testing.assert_array_equal(y, [1, 0])


@pytest.mark.parametrize("selected,expected", [
    (None, "default-target"), ("class", "class"),
])
def test_numeric_id_selects_target_and_uses_cache(fake_openml, selected, expected):
    calls, features, labels = fake_openml
    result = load_data(31, target=selected)
    assert calls == [dict(
        data_id=31, target_column=expected, as_frame=True, cache=True
    )]
    pd.testing.assert_frame_equal(result, pd.concat([labels, features], axis=1))


def test_integer_like_id_and_requested_target_name(fake_openml):
    calls, features, labels = fake_openml
    result = load_data(np.int64(42), target="outcome")
    assert calls[0]["data_id"] == 42
    assert type(calls[0]["data_id"]) is int
    assert calls[0]["target_column"] == "outcome"
    pd.testing.assert_frame_equal(result, pd.concat([
        labels.rename("outcome"), features
    ], axis=1))


def test_bundled_data_does_not_fetch_from_openml(fake_openml):
    calls, _, _ = fake_openml
    assert load_data().shape == (150, 5)
    assert load_data("artificial_tree").shape == (1440, 61)
    assert calls == []


@pytest.mark.parametrize("name", ["iris", "credit-g"])
def test_named_dataset_rejects_target_override(fake_openml, name):
    calls, _, _ = fake_openml
    with pytest.raises(ValueError, match="only be specified for an OpenML dataset ID"):
        load_data(name, target="other")
    assert calls == []


@pytest.mark.parametrize("name", [0, -1, True, 31.5, None])
def test_invalid_openml_ids_are_rejected(fake_openml, name):
    calls, _, _ = fake_openml
    with pytest.raises(ValueError, match="positive OpenML dataset ID"):
        load_data(name)
    assert calls == []


@pytest.mark.parametrize("target", ["", " ", 3, ["class"]])
def test_invalid_target_names_are_rejected(fake_openml, target):
    calls, _, _ = fake_openml
    with pytest.raises(ValueError, match="nonempty column name"):
        load_data(31, target=target)
    assert calls == []


def test_unknown_name_lists_available_examples(fake_openml):
    calls, _, _ = fake_openml
    with pytest.raises(NameError, match="Known names") as error:
        load_data("unknown")
    assert "credit-g" in str(error.value)
    assert "wine-quality-red" in str(error.value)
    assert calls == []


def test_id_without_default_target_explains_selection(monkeypatch):
    features = pd.DataFrame({"feature": [1, 2]})
    monkeypatch.setattr(
        import_module("rfphate.load_data"), "fetch_openml",
        lambda **kwargs: SimpleNamespace(data=features, target=None),
    )
    with pytest.raises(ValueError, match="no default target; pass target="):
        load_data(100)


def test_multiple_targets_require_explicit_selection(monkeypatch):
    features = pd.DataFrame({"feature": [1, 2]})
    targets = pd.DataFrame({"left": [0, 1], "right": [1, 0]})
    monkeypatch.setattr(
        import_module("rfphate.load_data"), "fetch_openml",
        lambda **kwargs: SimpleNamespace(data=features, target=targets),
    )
    with pytest.raises(ValueError, match="multiple targets; pass target="):
        load_data(100)


def test_one_column_target_frame_is_supported(monkeypatch):
    features = pd.DataFrame({"feature": [1, 2]})
    targets = pd.DataFrame({"class": ["a", "b"]})
    monkeypatch.setattr(
        import_module("rfphate.load_data"), "fetch_openml",
        lambda **kwargs: SimpleNamespace(data=features, target=targets),
    )
    pd.testing.assert_frame_equal(
        load_data(100), pd.concat([targets, features], axis=1)
    )


def test_non_tabular_openml_result_is_rejected(monkeypatch):
    monkeypatch.setattr(
        import_module("rfphate.load_data"), "fetch_openml",
        lambda **kwargs: SimpleNamespace(data=np.ones((2, 2)), target=pd.Series([0, 1])),
    )
    with pytest.raises(ValueError, match="did not provide tabular features"):
        load_data(100)
