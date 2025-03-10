import pandas as pd
import numpy as np

def dataprep(data: pd.DataFrame, label_col_idx: int = 0, transform: str = 'normalize', encoding: str = 'integer'):
    """
    Preprocesses a pandas DataFrame by normalizing or standardizing numerical features.
    Encodes categorical variables and extracts labels if specified.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the data to be preprocessed.

    label_col_idx : int, optional (default=0)
        The index of the column to be used as the label. If None, only features are returned.

    transform : {'normalize', 'standardize'}, optional (default='normalize')
        The transformation to apply:
        - 'normalize': Scales numerical variables to [0,1] range.
        - 'standardize': Standardizes numerical variables (zero mean, unit variance).

    encoding : {'integer', 'onehot'}, optional (default='integer')
        The encoding to apply to categorical variables:
        - 'integer': Encodes categorical variables as integers.
        - 'onehot': Encodes categorical variables using one
          hot encoding (creates dummy variables).

    Returns:
    --------
    np.ndarray or tuple
        - If label_col_idx is None, returns a NumPy array of processed features.
        - Otherwise, returns a tuple (features as NumPy array, labels as pandas Series).

    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4],
    ...     'feature2': [5, 6, 7, 8],
    ...     'label': ['A', 'B', 'A', 'B']
    ... })
    >>> x, y = dataprep(data)
    >>> x_standardized, y_standardized = dataprep(data, transform='standardize')
    """

    data = data.copy()

    # Identify categorical columns and encode them
    categorical_cols = data.select_dtypes(include=['object', 'int64']).columns

    if encoding == 'onehot':
        data = pd.get_dummies(data, columns=categorical_cols)
    elif encoding == 'integer':
        data[categorical_cols] = data[categorical_cols].apply(lambda col: pd.Categorical(col).codes)

    # Extract label column if specified
    y = None
    if label_col_idx is not None:
        label = data.columns[label_col_idx]
        y = data.pop(label)

    # Apply transformation
    if transform == 'standardize':
        x = data.apply(lambda col: (col - col.mean()) / col.std() if col.std() != 0 else col)
    elif transform == 'normalize':
        x = data.apply(lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else col)
    else:
        raise ValueError("Invalid transform. Choose 'normalize' or 'standardize'.")

    return (x.to_numpy(), y) if label_col_idx is not None else x.to_numpy()
