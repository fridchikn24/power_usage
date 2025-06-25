import pandas as pd

class DataPrepper:
    """

    Class to prepare and feature engineer data including but
    not limited to, adding lag variables, moving averages, 
    and date decomposition

    """

    def __init__(self, df):
        """
        Parameters:

        df: pandas dataframe containing our data
        """

        self.df = df

    def decompose_dates(self.df):
        self.df['date'] = pd.to_datetime(self.df['date'])

        self.df['Year'] = self.df['date'].dt.year
        self.df['Month'] = self.df['date'].dt.month
        self.df['Day_of_Week']=self.df['date'].dt.dayofweek
        self.df['Is_Weekend'] = self.df['Day_of_Week'] >= 5
    
        return self.df