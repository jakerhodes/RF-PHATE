import pandas as pd
import numpy as np

def dataprep(data, label_col_idx = 0, transform = 'normalize'):
    
    """
    Reads in a pandas dataframe and returns a normalized or standardized numpy array along with corresponding labels (if provided in the dataframe). The column label must be identified via the 'label_col_idx' argument.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the data to be preprocessed.

    label_col_idx : int, optional, default: 0
        The index of the column to be used as the label. If None, only the features will be returned.

    transform : {'normalize', 'standardize'}, optional, default: 'normalize'
        The type of transformation to apply. 'normalize' scales categorical variables from 0 to 1,
        assigning the highest value the value of 1 and the lowest value the value of 0.
        'standardize' standardizes numerical variables by subtracting the mean and dividing by the standard deviation.

    Returns:
    --------
    np.ndarray or tuple
        If label_col_idx is None, returns a NumPy array containing the preprocessed features.
        If label_col_idx is specified, returns a tuple containing a NumPy array of features and the corresponding labels (as a pandas series).

    Examples:
    ---------
    >>> import pandas as pd
    >>> import numpy as np

    >>> # Creating a sample DataFrame
    >>> data = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4],
    ...     'feature2': [5, 6, 7, 8],
    ...     'label': ['A', 'B', 'A', 'B']
    ... })

    >>> # Preprocessing the data with default parameters
    >>> x, y = dataprep(data)

    >>> # Preprocessing the data with standardization
    >>> x_standardized, y_standardized = dataprep(data, transform = 'standardize')
    """

    data = data.copy()
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' or data[col].dtype == 'int64']
    
    for col in categorical_cols:
        data[col] = pd.Categorical(data[col]).codes

    if label_col_idx is not None:
        label = data.columns[label_col_idx]
        y = data.pop(label)
        x = data
    else:
        x = data

    if transform == 'standardize':
        for col in x.columns:
            if x[col].std() != 0:
                x[col] = (x[col] - x[col].mean()) / x[col].std()
    elif transform == 'normalize':
        for col in x.columns:
            if x[col].max() != x[col].min():
                x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())

    if label_col_idx is None:
        return x.to_numpy()
    else:
        return x.to_numpy(), y
