README - RF-PHATE
=================

RF-PHATE is an algorithm that allows the user to create random forest-based supervised, low-dimensional embeddings based on the manifold learning algorithm described in "Gaining Biological Insights through Supervised Data Visualization" [1]_.

Documentation
-------------

For documentation, please visit `ReadTheDocs: RF-PHATE <https://jakerhodes.github.io/RF-PHATE/>`_.

Installation and updating
-------------------------

Use the package manager `pip` to install RF-PHATE. Rerun this command to check for and install updates. Installation should take no more than 5 minutes. The package requires `python>=3.7`.

.. code-block:: bash

    pip install git+https://github.com/jakerhodes/RF-PHATE

The random forest implementation is based on either `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_ or `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ from `scikit-learn <https://scikit-learn.org/stable/>`_, depending on the type of response variable (categorical or continuous). The user implicitly chooses the type of random forest model by including either the response variable, `y`, or by stating the `prediction_type` as either 'classification' or 'regression'. If neither `y` nor `prediction_type` is given, RF-PHATE will assume a categorical response. Any training options available for `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_ or `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ will work for the RF-PHATE initialization.

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

For code to generate the quantification results from the paper, please see
`RF-PHATE-Quantification, <https://github.com/jakerhodes/RF-PHATE-Quantification>`_

Quantification Results
----------------------
If you find the RF-PHATE method useful, please cite:

References
----------
.. [1] 
    Rhodes, J.S., Aumon, A., Morin, S., et al.: Gaining Biological Insights through Supervised
    Data Visualization. bioRxiv (2023). https://doi.org/10.1101/2023.11.22.568384.
