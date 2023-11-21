RF-PHATE
========

RF-PHATE is an algorithm that allows the user to create random forest-based supervised, low-dimensional embeddings based on the manifold learning algorithm described in "Random Forest-Based Diffusion Information Geometry for Supervised Visualization and Data Exploration" [1]_.

Documentation
-------------

For documentation, please visit `ReadTheDocs: RF-PHATE <https://jakerhodes.github.io/RF-PHATE/>`_.

Installation and updating
-------------------------

Use the package manager `pip` to install RF-PHATE. Rerun this command to check for and install updates.

.. code-block:: bash

    pip install git+https://github.com/jakerhodes/RF-PHATE

The random forest implementation is based on either `RandomForestClassifier` [2]_ or `RandomForestRegressor` [3]_ from `scikit-learn` [4]_, depending on the type of response variable (categorical or continuous). The user implicitly chooses the type of random forest model by including either the response variable, `y`, or by stating the `prediction_type` as either 'classification' or 'regression'. If neither `y` nor `prediction_type` is given, RF-PHATE will assume a categorical response. Any training options available for `RandomForestClassifier` [2]_ or `RandomForestRegressor` [3]_ will work for the RF-PHATE initialization.

Quick Demo
----------

Below is a quick demo of how to use RF-PHATE on the Titanic dataset:

.. code-block:: python

    from rfphate import RFPHATE
    from dataprep import dataprep
    import pandas as pd
    import seaborn as sns

    data = pd.read_csv('datasets/titanic.csv')
    x, y = dataprep(data)

    rfphate = RFPHATE(y=y, random_state=0)
    # Alternatively, rfphate = RFPHATE(prediction_type='classification', random_state=0)

    embedding = rfphate.fit_transform(x, y)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=pd.Categorical(data.iloc[:, 0]))

.. image:: figures/titanic.png

We can visually explore the relationships between the response (survival) and other feature variables:

By passenger class:

.. code-block:: python

    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=pd.Categorical(data.iloc[:, 1]))
    plt.legend(title='By Class')

.. image:: figures/titanic_class.png

By passenger sex:

.. code-block:: python

    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=pd.Categorical(data.iloc[:, 2]))
    plt.legend(title='By Sex')

.. image:: figures/titanic_sex.png

If you find the RF-PHATE method useful, please cite:

J. S. Rhodes, A. Cutler, G. Wolf, and K. R. Moon, "Random Forest-Based Diffusion Information Geometry for Supervised Visualization and Data Exploration," 2021 IEEE Statistical Signal Processing Workshop (SSP), Rio de Janeiro, Brazil, 2021, pp. 331-335, doi: 10.1109/SSP49050.2021.9513749.
