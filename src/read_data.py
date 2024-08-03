import os 
import pandas as pd

class ReadData:
    """This class loads and preprocesses the data.
    
    First the data is loaded from Excel and converted to CSV. This is done only once. Next some global preprocessing of 
    the input dataframe is applied. Features of the data, such as time_series, are calculated on request and saved in a 
    library.
    
    Attributes:
        data (pd.DataFrame): parsed input dataset
        data_library (dict): contains many different data structures, such as time series
    """

    def __init__(self, run_sett: dict):
        """constructs a ReadData object"""
        self._run_sett = run_sett
        #input_path = os.path.join(run_sett['input_dir'], run_sett['input_fn'])
        #raw_data = pd.read_csv(input_path + '.csv', parse_dates=['Period'])
        #self.data = self.parse_data(data=raw_data)
        self.data_library = {}
    
    def parse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies some global cleaning of the data.
        
        Error fixes:
            ...
            ...
            ...

        Args:
            data: raw input data

        Returns:
            data: parsed input data
        """
        pass