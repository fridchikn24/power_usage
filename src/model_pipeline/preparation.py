import pandas as pd
from loguru import logger
import holidays


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

    def decompose_dates(self,df):

        logger.info("decomposing datetime")
        df['date'] = pd.to_datetime(df['date'])

        df['Year'] = df['date'].dt.year
        df['Month'] = df['date'].dt.month
        df['Day_of_Week']=df['date'].dt.dayofweek
        df['Is_Weekend'] = df['Day_of_Week'] >= 5
    
    def identify_holidays(self, df):
       
        logger.info("Checking for holidays")
        us_holidays = holidays.US(years=[2022,2023,2024])
        # Check if each date is a holiday
        df['Is_Holiday'] = df['date'].isin(us_holidays)

    def create_lag(self,df):

        """
        function to create lagged y variables based on the "active power" variable
        """

        logger.info('creating lag variables')
        for lag in [1, 2, 5,7]:
            df[f'lag_{lag}'] = df['active_power'].shift(lag)

    def create_moving_averages(self,df):
        """
        create moving averages and standard deviation 
        for the active power variable
        """
        for window in [3, 7]:
            df[f'roll_mean_{window}'] = df['active_power'].shift(1).rolling(window=window).mean()
            df[f'roll_std_{window}'] = df['active_power'].shift(1).rolling(window=window).std()


    def transform(self):
        """
        Runs all transformation steps in sequence and returns the final engineered DataFrame.
        """
        df = self.df.copy()

        self.decompose_dates(df)
        self.identify_holidays(df)
        self.create_lag(df)
        self.create_moving_averages(df)

        df = df.dropna().reset_index(drop=True)  # Remove NaNs from lag/rolling ops
        return df