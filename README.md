# RF-PHATE

RF-PHATE is an algorithm that allows the user to create random forest-based supervised, low-dimensional embeddings based on the 
manifold learning algorithm described in 
[Random Forest-Based Diffusion Information Geometry for Supervised Visualization and Data Exploration](https://ieeexplore.ieee.org/document/9513749).

## Installation and updating
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install RF-PHATE.
Rerun this command to check for and install  updates.
```bash
pip install git+https://github.com/jakerhodes/RFPHATE
```

The random forest implementation is based on either [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) from [scikit-learn](https://scikit-learn.org/stable/), depending on the type of response variable (categorical or continuous). The user implicitly chooses the type of random forest model by including either the response variable, $y$, or by stating the *prediction_type* as either 'classification' or 'regression'. If neither $y$ nor *prediction_type* is given, RF-PHATE will assume a categorical response. Any training options available for [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) will work for the RF-PHATE initialization.

Below is a quick demo of how to use RF-PHATE on the Titanic dataset:

```python
from rfphate import RFPHATE
from dataprep import dataprep
import pandas as pd
import seaborn as sns

data = pd.read_csv('datasets/titanic.csv')
x, y = dataprep(data)

rfphate = RFPHATE(y = y, random_state = 0)
# Alternatively, rfphate = RFPHATE(prediction_type = 'classification', random_state = 0)

embedding = rfphate.fit_transform(x, y)
sns.scatterplot(x = embedding[:, 0], y = embedding[:, 1], hue = pd.Categorical(data.iloc[:, 0]))

```
![](figures/titanic.png)

We can visually explore the relationships between the response (survival) and other feature variables:

By passenger class:
```python
sns.scatterplot(x = embedding[:, 0], y = embedding[:, 1], hue = pd.Categorical(data.iloc[:, 1]))
plt.legend(title = 'By Class')
```
![](figures/titanic_class.png)


By passenger sex:
```python
sns.scatterplot(x = embedding[:, 0], y = embedding[:, 1], hue = pd.Categorical(data.iloc[:, 2]))
plt.legend(title = 'By Sex')
```
![](figures/titanic_sex.png)

If you find the RF-PHATE method useful, please cite:

J. S. Rhodes, A. Cutler, G. Wolf and K. R. Moon, "Random Forest-Based Diffusion Information Geometry for Supervised Visualization and Data Exploration," 2021 IEEE Statistical Signal Processing Workshop (SSP), Rio de Janeiro, Brazil, 2021, pp. 331-335, doi: 10.1109/SSP49050.2021.9513749.
