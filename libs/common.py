import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Union


def is_datetimeindex(func):
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        
        # check if dataframe has datetime index
        assert isinstance(df.index, pd.DatetimeIndex), 'Dataframe index is not of type `pd.DateTimeIndex`'

        return func(df, *args, **kwargs)
    return wrapper

def is_reindexed(func):
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        
        # check if dataframe has datetime index
        assert pd.infer_freq(df.index)=='D', 'Dataframe not indexed correctly'

        return func(df, *args, **kwargs)
    return wrapper


@is_datetimeindex
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['dow'] = df.index.dayofweek + 1 # since dayofweek is 0-indexed
    df['woy'] = df.index.isocalendar().week
    df['doy'] = df.index.day_of_year
    df['moy'] = df.index.month
    df['qtr'] = df.index.quarter

    # Define periodicity of time components
    components = {
        'dow': 7,
        'woy': 52,
        'doy': 365,  # assuming that every year has 365 days
        'moy': 12,
        'qtr': 4
        }

    for component, periodicity in components.items():
        sine = 'sin_'+component
        cosine = 'cos_'+component

        df[sine] = np.sin(2 * np.pi * df[component]/periodicity)
        df[cosine] = np.cos(2 * np.pi * df[component]/periodicity)
    
    return df


@is_datetimeindex
def define_holidays(df: pd.DataFrame) -> pd.DataFrame:
    
    # Weekend
    df['is_weekend'] = df.index.dayofweek > 4
    df['is_weekend'] = df['is_weekend'].astype(int)

    # National holidays
    cal: pd.DataFrame = pd.read_csv('./data/can_holiday_calendar.csv', parse_dates=['date'], index_col='date')
    df = pd.merge(df, cal, how='left', left_index=True, right_index=True, sort=False, copy=False)
    df['is_nat_holiday'] = np.where(df['holiday'].notnull(), 1, 0)

    # Combine weekends and national holidays
    cond = (df['is_weekend'] | df['is_nat_holiday'])
    df['is_holiday'] = np.where(cond, 1, 0)

    df.drop(['holiday'], axis='columns', inplace=True)

    return df


@is_datetimeindex
def holidays_in_N_days(df: pd.DataFrame, N: int) -> pd.DataFrame:

    # check if 'is_holiday' columns exists
    assert 'is_holiday' in df.columns, 'is_holiday feature is not present'
    
    _colname = f'holidays_in_last_{str(N)}_days'
    window_size = f'{N}D'

    df[_colname] = df['is_holiday'].rolling(window_size, closed='left').sum()

    return df


@is_datetimeindex
def create_lags(df: pd.DataFrame, lag_column: str, lag: int, frequency:str='D') -> pd.DataFrame:

    _colname = f't-{lag}{frequency}'
    df[_colname] = df[lag_column].shift(periods=lag, freq=frequency)

    return df


@is_datetimeindex
def rolling_mean(df: pd.DataFrame, target_column: str, window_size: int) -> pd.DataFrame:

    _colname = f'{window_size}_day_rolling_mean'
    _window_size = f'{window_size}D'

    df[_colname] = df[target_column].rolling(_window_size, closed='left').mean()

    return df


@is_datetimeindex
@is_reindexed
def percentage_change(df: pd.DataFrame, target_column: str, frequency: str) -> pd.DataFrame:
    
    _colname = f'{frequency}_pct_change'

    df[_colname] = df[target_column].pct_change(freq=frequency)

    return df


@is_datetimeindex
def reindex_by_date(df: pd.DataFrame, min_date: datetime, max_date:datetime, ffill_col: str) -> pd.DataFrame:

    date_range = pd.date_range(min_date, max_date, freq='D')
    df = df.reindex(date_range)

    fill_value = df[ffill_col].mode()
    print(fill_value)
    df[ffill_col] = fill_value
    # df.fillna(0, inplace=True)

    return df


@is_datetimeindex
@is_reindexed
def momentum(df: pd.DataFrame, target_column: str, num_days: int) -> pd.DataFrame:

    _prev_day = df[target_column].shift(periods=1, freq='D')
    _N_days_ago = df[target_column].shift(periods=num_days, freq='D')

    _col_name = f'M{num_days}'

    df[_col_name] = _prev_day.subtract(_N_days_ago)

    return df


@is_datetimeindex
@is_reindexed
def MACD(df: pd.DataFrame, target_column: str, low: int, high: int) -> pd.DataFrame:

    _EMA_low: pd.Series = df[target_column].ewm(span=low, adjust=False).mean()
    _EMA_high: pd.Series = df[target_column].ewm(span=high, adjust=False).mean()

    _col_name = f'{low}_{high}_MACD'

    df[_col_name] = _EMA_low.subtract(_EMA_high)

    return df
