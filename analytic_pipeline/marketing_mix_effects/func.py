import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class AdstockGeometric(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        x_decayed = np.zeros_like(X)
        x_decayed[0] = X[0]

        for xi in range(1, len(x_decayed)):
            x_decayed[xi] = X[xi] + self.alpha * x_decayed[xi - 1]
        return x_decayed


class Report:
    @staticmethod
    def effects(df_shap_values, media_channels, df_original):
        """
        Args:
            df_shap_values: data frame of shap values
            media_channels: list of media channel names
            df_original: non-transformed original data
        Returns:
            pd.DataFrame: data frame with spend effect shares
        """
        # Effects:
        responses = pd.DataFrame(df_shap_values[media_channels].abs().sum(), columns=["effect_share"])
        response_percentages = responses / responses.sum()

        # Shares:
        tmp = df_original[media_channels].sum() / df_original[media_channels].sum().sum()
        spends_percentages = pd.DataFrame(tmp, columns=["spend_share"])

        # Merge
        spend_effect_share = pd.merge(response_percentages, spends_percentages, left_index=True, right_index=True)
        spend_effect_share = spend_effect_share.reset_index().rename(columns={"index": "media"})
        return spend_effect_share

    @staticmethod
    def rssd(effect_share, spend_share):
        """
        Root-sum-square distance, a major innovation of Robyn
        eliminates the majority of "bad models"
        (larger prediction error and/or unrealistic media effect like the smallest channel getting the most effect
        Args:
            effect_share ([type]): percentage of effect share
            spend_share ([type]): percentage of spend share
        Returns:
            [type]: [description]
        """
        return np.sqrt(np.sum((effect_share - spend_share) ** 2))


class Data:
    def __init__(self, path):
        self.path = path

    def holiday(self, file_name):
        holidays = pd.read_csv(self.path / file_name, parse_dates=["ds"])
        holidays["begin_week"] = holidays["ds"].dt.to_period('W-SUN').dt.start_time
        holidays_weekly = holidays.groupby(["begin_week", "country", "year"], as_index=False).agg(
            {'holiday': '#'.join, 'country': 'first', 'year': 'first'}).rename(columns={'begin_week': 'ds'})
        holidays_weekly_de = holidays_weekly.query("(country == 'DE')").copy()
        return holidays_weekly_de

    @staticmethod
    def seasonal_prophet(df_holiday, prophet_data):
        prophet = Prophet(yearly_seasonality=True, holidays=df_holiday)
        prophet.add_regressor(name='events_event2')
        prophet.add_regressor(name='events_na')
        prophet.fit(prophet_data[["ds", "y", "events_event2", "events_na"]])
        return prophet.predict(prophet_data[["ds", "y", "events_event2", "events_na"]])
