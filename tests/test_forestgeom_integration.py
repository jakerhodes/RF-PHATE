import unittest

import numpy as np
from forestgeom import Proximity
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
    GradientBoostingClassifier,
)

from rfphate import RFPHATE


class ForestgeomIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_classification(
            n_samples=60, n_features=6, n_informative=4, random_state=42
        )

    def test_default_forest(self):
        model = RFPHATE(random_state=17, n_jobs=2)
        forest = model._make_proximity_model().forest
        self.assertIsInstance(forest, RandomForestClassifier)
        self.assertEqual(forest.get_params(), RandomForestClassifier().get_params())

    def test_supplied_forest_settings_and_fitted_reuse(self):
        forest = RandomForestClassifier(
            n_estimators=40, max_depth=3, random_state=7, n_jobs=1
        )
        model = RFPHATE(forest=forest, random_state=17, n_jobs=2)
        proximity = model._make_proximity_model()
        self.assertIs(proximity.forest, forest)
        proximity.fit(self.x, self.y)
        self.assertFalse(hasattr(forest, "estimators_"))
        self.assertEqual(proximity.forest_.estimator.get_params(), forest.get_params())
        forest.fit(self.x, self.y)
        fitted_proximity = model._make_proximity_model().fit(self.x, self.y)
        self.assertIs(fitted_proximity.forest_.estimator, forest)

    def test_embedding_and_extension(self):
        for scheme, estimator, self_similarity, n_landmark in (
            ("gap", RandomForestClassifier, False, None),
            ("gap", RandomForestClassifier, True, None),
            ("uniform", ExtraTreesClassifier, False, None),
            ("oob", RandomForestClassifier, False, None),
            ("kerf", RandomForestClassifier, False, None),
            ("boosted", GradientBoostingClassifier, False, None),
            ("gap", RandomForestClassifier, False, 10),
            ("gap", RandomForestRegressor, False, None),
        ):
            with self.subTest(
                scheme=scheme, estimator=estimator.__name__,
                self_similarity=self_similarity, n_landmark=n_landmark,
            ):
                model = RFPHATE(
                    forest=estimator(
                        n_estimators=40, max_depth=4, random_state=42
                    ),
                    random_state=42,
                    self_similarity=self_similarity,
                    proximity_params={"weight_scheme": scheme},
                    phate_params={
                        "t": 2, "n_landmark": n_landmark, "n_svd": 5,
                        "mds": "classic", "verbose": 0,
                    },
                )
                embedding = model.fit_transform(
                    self.x, self.y, force_symmetric=True
                )
                self.assertIsInstance(model.proximity_model_, Proximity)
                self.assertEqual(embedding.shape, (60, 2))
                self.assertTrue(np.isfinite(embedding).all())
                self.assertTrue(sparse.issparse(
                    model.proximity_model_.training_proximity()
                ))
                new_x = self.x[:5] + 0.01
                transitions = model.extend_to_data(new_x)
                np.testing.assert_allclose(
                    np.asarray(transitions.sum(axis=1)).ravel(), 1, atol=1e-6
                )
                projected = model.transform(new_x)
                self.assertEqual(projected.shape, (5, 2))
                self.assertTrue(np.isfinite(projected).all())


if __name__ == "__main__":
    unittest.main()
