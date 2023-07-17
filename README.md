# RF-PHATE-Official
Official RF-PHATE repository. To be renamed RF-PHATE after completion, old repo to be RF-PHATE-Experiments


```python

from rfphate import RFPHATE
from dataprep import dataprep
import pandas as pd
import seaborn as sns



data = pd.read_csv('.datasets/titanic.csv')
x, y = dataprep(data)

rfphate = RFPHATE(y = y, random_state = 0)
embedding = rfphate.fit_transfomr(x)

sns.scatterplot(x = embedding[:, 0], y = embedding[:, 1], hue = pd.Categorical(data.iloc[:, 0]))

```