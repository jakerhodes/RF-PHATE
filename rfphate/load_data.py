import pkg_resources
import pandas as pd


def load_data(name = 'iris'):
    """
    Function to load data from the rfphate package.
    Data is read in as a pandas DataFrame. In each example,
    the first column of the dataframe is the data's labels.

    Parameters
    ----------
    name : string
    Options are 'iris', 'titanic', 'auto-mpg'

    Returns
    -------
    A pandas dataframe of the requested data

    """

    if name == 'iris':
        path = 'datasets/iris.csv'

    elif name == 'titanic':
        path = 'datasets/titanic.csv'

    elif name == 'auto-mpg':
        path = 'datasets/auto-mpg.csv'
    else:
        raise NameError("{} is not recognized. The only names are 'iris', 'titanic', and 'auto-mpg'.".format(name))
    

    data_path = pkg_resources.resource_filename('rfphate', path)
    
    return pd.read_csv(data_path)