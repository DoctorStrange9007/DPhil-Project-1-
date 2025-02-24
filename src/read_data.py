import os
import pandas as pd
import numpy as np
from statsmodels.api import OLS
from statsmodels.api import add_constant


class ReadData:
    """Class for loading and preprocessing financial time series data.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing runtime settings including:
        - years : list[int]
            Years of data to process
        - input_dir : str
            Directory path containing the input data files

    Attributes
    ----------
    data : pd.DataFrame
        Processed and cleaned dataset
    _raw_data : pd.DataFrame
        Raw dataset before cleaning

    Notes
    -----
    The data loading process follows these steps:
    1. Loads data from compressed CSV files organized by year
    2. Calculates log returns from price data
    3. Applies data cleaning to remove invalid entries
    """

    def __init__(self, run_sett: dict):
        """constructs a ReadData object"""
        self._run_sett = run_sett
        self._raw_data = self.read_raw()
        self._data = self.parse_data()
        self.rolling_train_test_splits = self.get_rolling_train_test_splits(
            L=self._run_sett["data"]["L"], S=self._run_sett["data"]["S"]
        )

    def read_raw(self):
        """Load and combine raw financial data across multiple years.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame containing log returns for all years
            Index: ticker symbols
            Columns: dates in format YYYYMMDD

        Notes
        -----
        The process:
        1. Iterates through specified years
        2. Loads and processes each year's data
        3. Concatenates all years together
        4. Keeps only rows/columns present in all years (inner join)
        """
        years = self._run_sett["years"]
        raw_data = self.read_raw_year(years[0])

        for year in years[1:]:
            raw_data_temp = self.read_raw_year(year)
            raw_data = pd.concat([raw_data, raw_data_temp], axis=1, join="inner")

        return raw_data.iloc[
            :50
        ]  # for now due to computational time XXXXXXXCHANGEXXXXXXX

    def read_raw_year(self, year):
        """Load and process financial data for a specific year.

        Parameters
        ----------
        year : int
            Year to process

        Returns
        -------
        pd.DataFrame
            DataFrame containing log returns for the specified year
            Index: ticker symbols
            Columns: dates in format YYYYMMDD

        Notes
        -----
        The process:
        1. Loads daily price data from compressed CSV files
        2. Extracts closing prices
        3. Calculates log returns: r_t = log(p_t/p_{t-1})
        4. Removes the first day (no return calculable)
        """
        sorted_strings = sorted(
            list_visible_files_with_list_comprehension(
                self._run_sett["input_dir"] + str(year)
            ),
            key=lambda x: int(x[:8]),
        )
        input_path_init = os.path.join(
            self._run_sett["input_dir"] + str(year), sorted_strings[0]
        )
        raw_data_year = (
            pd.read_csv(input_path_init, index_col="ticker", compression="gzip")
            .loc[:, "close"]
            .rename(sorted_strings[0][:8])
            .to_frame()
        )

        for day_file in sorted_strings[1:]:
            input_path = os.path.join(self._run_sett["input_dir"] + str(year), day_file)
            raw_data_year_temp = (
                pd.read_csv(input_path, index_col="ticker", compression="gzip")
                .loc[:, "close"]
                .rename(day_file[:8])
            )
            raw_data_year = pd.concat(
                [raw_data_year, raw_data_year_temp], axis=1, join="inner"
            )

        raw_data_year = (
            np.log(raw_data_year.transpose() / raw_data_year.transpose().shift(1))
            .iloc[1:,]
            .transpose()
        )

        return raw_data_year

    def parse_data(self) -> pd.DataFrame:
        """Clean the raw financial data by removing invalid entries.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset with invalid data removed

        Notes
        -----
        Cleaning steps:
        1. Removes days where >10% of stocks have zero returns
        2. Removes stocks where >50% of days have zero returns
        3. Removes days where >10% of stocks have returns >100%
        4. Removes stocks where >50% of days have returns >100%
        """
        data = self._raw_data
        thresh_n = data.shape[0] // 10
        thresh_d = data.shape[1] // 2
        zero_return_df = data == 0
        bigger_than_one_return_df = data > 1

        valid_days = ~(
            (zero_return_df.sum() > thresh_n)
            | (bigger_than_one_return_df.sum() > thresh_n)
        )
        valid_stocks = ~(
            (zero_return_df.sum(axis=1) > thresh_d)
            | (bigger_than_one_return_df.sum(axis=1) > thresh_d)
        )
        parsed_data = data.loc[valid_stocks, valid_days]
        non_company = self._run_sett["non_company"]
        SPY_ETF = parsed_data.loc["SPY", :]
        parsed_data = parsed_data.loc[~parsed_data.index.isin(non_company)]
        parsed_data = self.calculate_excess_returns(parsed_data, SPY_ETF)
        parsed_data = parsed_data.apply(pd.to_numeric)

        return parsed_data

    def calculate_excess_returns(
        self, data: pd.DataFrame, SPY_ETF: pd.Series, window: int = 60
    ) -> pd.DataFrame:
        """Calculate excess returns from price data, where Beta is calculated on a rolling 60 day window.

        Parameters
        ----------
        data : pd.DataFrame
            Price data

        Returns
        -------
        pd.DataFrame
            Excess returns"""
        dates = data.columns

        date_sets_per_window = [
            dates[i : i + window] for i in range(0, len(dates), window)
        ]
        all_dates_except_last_set = dates[~dates.isin(date_sets_per_window[0])]
        test_data_sets = [
            data.loc[:, dates.isin(date_set)] for date_set in date_sets_per_window
        ]
        SPY_ETF_sets = [
            SPY_ETF.loc[dates.isin(date_set)] for date_set in date_sets_per_window
        ]

        excess_returns_df = pd.DataFrame(
            index=data.index, columns=all_dates_except_last_set
        )

        # Iterate over each company
        for company in data.index:
            # Perform regression for each window
            for i, (test_data, spy_etf) in enumerate(
                zip(test_data_sets[:-1], SPY_ETF_sets[:-1])
            ):
                y = test_data.loc[company]
                X = spy_etf
                X = add_constant(X)  # Add constant for intercept

                # Fit the model
                model = OLS(y, X).fit()
                beta = model.params[1]  # Get the beta coefficient

                next_window_data = test_data_sets[i + 1].loc[company]
                excess_return = next_window_data - beta * SPY_ETF_sets[i + 1]

                # Store the excess returns in the DataFrame
                excess_returns_df.loc[company, next_window_data.index] = excess_return

        return excess_returns_df

    def get_rolling_train_test_splits(self, L: int, S: int):
        """Perform rolling window training and testing using parallel processing.

        Parameters
        ----------
        L : int
            Lookback window size (e.g., 252 trading days)
        S : int
            Step size for model refitting (e.g., 21 trading days)
        all_data : pd.DataFrame
            Complete dataset for training and testing

        Returns
        -------
        tuple
            - train_data_sets : list of pd.DataFrame train sets
            - test_data_sets : list of pd.DataFrame test sets
            - forecasted_dates : pandas.DatetimeIndex
                Dates corresponding to the forecasts
        """
        all_data = self._data
        nr_models = all_data.shape[1] // S
        dates = pd.to_datetime(all_data.columns)

        train_date_sets = [dates[i : i + L] for i in range(0, len(dates) - L, S)][
            :nr_models
        ]
        train_data_sets = [
            all_data.loc[:, dates.isin(date_set)] for date_set in train_date_sets
        ]

        test_date_sets = [dates[i : i + S] for i in range(L - 1, len(dates), S)]
        forecasted_dates = dates[L:]
        test_data_sets = [
            all_data.loc[:, dates.isin(date_set)] for date_set in test_date_sets
        ]

        return train_data_sets, test_data_sets, forecasted_dates


def list_visible_files_with_list_comprehension(directory):
    """List all non-hidden files in a directory.

    Parameters
    ----------
    directory : str
        Path to the directory to scan

    Returns
    -------
    list
        List of filenames (excluding those starting with '.')
    """
    visible_files = [file for file in os.listdir(directory) if not file.startswith(".")]
    return visible_files
