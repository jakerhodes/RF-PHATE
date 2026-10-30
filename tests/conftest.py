import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from rfphate import RFPHATE


@pytest.fixture(scope="module")
def classification_data():
    return make_classification(
        n_samples=80, n_features=6, n_informative=4, random_state=42
    )


@pytest.fixture
def make_model():
    def factory(**kwargs):
        options = {
            "forest": RandomForestClassifier(
                n_estimators=40, max_depth=4, random_state=42, n_jobs=1
            ),
            "random_state": 42,
            "phate_params": {
                "t": 2, "n_landmark": None, "n_svd": 5,
                "mds": "classic", "verbose": 0,
            },
        }
        options.update(kwargs)
        return RFPHATE(**options)
    return factory
