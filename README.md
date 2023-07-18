# RF-PHATE

RF-PHATE is an algorithm which allows the user to create random forest-based supervised, low-dimensional embeddings based on the 
manifold learning algorithm described in 
[Random Forest-Based Diffusion Information Geometry for Supervised Visualization and Data Exploration](https://ieeexplore.ieee.org/document/9513749)

## Installation and updating
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install RF-PHATE.
Rerun this command to check for and install  updates .
```bash
pip install git+https://github.com/jakerhodes/RF-PHATE-Official
```

The random forest implementation is based on either RandomForestClassifier or RandomForestRegressor from [scikit-learn](https://scikit-learn.org/stable/), depending on the type of response variable (categorical or continuous). The user implicitly chooses the type of random forst model by including either the response variable, $y$, or by stating the *prediction_type* as either 'classification' or 'regression'. If neither $y$ or *prediction_type* is given, RF-PHATE will assume a categorical response.

Below is a quick demo how to use RF-PHATE:

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