import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from optuna import logging
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

plt.style.use('ggplot')
logging.set_verbosity(logging.WARNING)


class Training:
    def __init__(self, all_features, label_name, data):
        self.all_features = all_features
        self.label_name = label_name
        self.data = data
        self.dataset = None
        self.lgb_data = None

    def train_split(self, date_from, date_to):
        date_from, date_to = pl.lit(date_from).str.strptime(pl.Date), pl.lit(date_to).str.strptime(pl.Date)
        self.dataset = {
            'train': self.data.filter(pl.col('grass_date') < date_from),
            'valid': self.data.filter(pl.col('grass_date').is_between(date_from, date_to)),
            'test': self.data.filter(pl.col('grass_date') > date_to),
        }
        self.lgb_data = {
            i: lgb.Dataset(self.dataset[i][self.all_features].to_numpy(), self.dataset[i][self.label_name].to_numpy())
            for i in self.dataset
        }
        return self.lgb_data, self.dataset

    def results(self, model, col_key: list):
        results = {}
        for name in tqdm(self.dataset, desc='Report prediction: '):
            pred = np.rint(model.predict(self.dataset[name][self.all_features].to_numpy(), num_iteration=model.best_iteration))

            tmp = (
                self.dataset[name].select(pl.col(col_key))
                .with_columns(
                    pl.col('grass_date').dt.strftime(format="%Y-%m-01").alias('grass_month'),
                    pl.Series(name='predictions', values=pred),
                    pl.Series(name='true_label', values=self.dataset[name][self.label_name]),
                )
            )
            tmp = tmp.with_columns(
                (pl.col('true_label') / pl.col('predictions') - 1).alias('mape'),
                pl.lit(name).alias('dataset')
            )
            results[name] = tmp
        return results

    def feature_importance(self, model):
        zip_ = zip(self.all_features, model.feature_importance())
        return (
            pl.DataFrame(zip_, schema=['feature', '# times the feature is used'])
            .sort('# times the feature is used', descending=True)
        )

    def plot(self, data, figname):
        fig, axes = plt.subplots(1, len(data), figsize=(25, 6))
        axes = axes.flatten()
        for idx, (name, df) in enumerate(data.items()):
            df[['grass_date', 'predictions', 'true_label']].to_pandas().set_index('grass_date').plot(kind='line', ax=axes[idx])
            mae = mean_absolute_error(df['true_label'], df['predictions'])
            mape = mean_absolute_percentage_error(df['true_label'], df['predictions'])
            axes[idx].set_title(f'{figname.stem}\n{name} - MAE: {mae:,.0f} - MAPE: {mape:,.2f}')
        fig.tight_layout()
        fig.savefig(figname, dpi=300)


def objective(trial, lgb_data, all_features, label_name, dataset):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'seed': 42,
        'verbose': -1,
        'feature_pre_filter': False,
        'deterministic': True,
        'num_iteration': trial.suggest_int('num_iteration', 100, 8000),
        # 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    # Add a callback for pruning.
    model = lgb.train(param, lgb_data['train'],
                      valid_sets=[lgb_data['valid']],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    # Prediction
    preds = model.predict(dataset['test'][all_features].to_numpy(), num_iteration=model.best_iteration)
    rmse = mean_squared_error(dataset['test'][label_name], preds, squared=True)
    # Save model
    model.save_model(InputID.path.parent / f'models/{label_name}_{trial.number}.pkl')
    return rmse


class Extract:
    @staticmethod
    def month_day(df, col):
        return df.with_columns(
            pl.col(col).dt.year().alias('year').cast(pl.Int16),
            pl.col(col).dt.month().alias('month').cast(pl.Int8),
            pl.col(col).dt.day().alias('day').cast(pl.Int8),
            pl.col(col).dt.hour().alias('hour').cast(pl.Int8),
            pl.col(col).dt.minute().alias('minute').cast(pl.Int8),
            pl.col(col).dt.second().alias('second').cast(pl.Int8),
        )

    @staticmethod
    def cycle_time(df):
        return df.with_columns(
            pl.col('month').map(lambda x: np.sin(2 * np.pi * x / 12)).alias('month_sin'),
            pl.col('month').map(lambda x: np.cos(2 * np.pi * x / 12)).alias('month_cos'),
            pl.col('day').map(lambda x: np.sin(2 * np.pi * x / 31)).alias('day_sin'),
            pl.col('day').map(lambda x: np.cos(2 * np.pi * x / 31)).alias('day_cos'),
            pl.col('hour').map(lambda x: np.sin(2 * np.pi * x / 24.0)).alias('hour_sin'),
            pl.col('hour').map(lambda x: np.cos(2 * np.pi * x / 24.0)).alias('cos_sin'),
        )

    @staticmethod
    def trend(df, col, window=7):
        return df.with_columns(
            pl.col(i).rolling_mean(window).alias(f'trend_{window}d_{i}') for i in col
        )

    @staticmethod
    def season(df, col, window=7):
        return df.with_columns(
            (pl.col(i) - pl.col(f'trend_{window}d_{i}')).alias(f'season_{window}d_{i}') for i in col
        )

    @staticmethod
    def lag(df, col, window=7):
        return df.with_columns(
            pl.col(i).shift(window).alias(f'shift_{window}d_{i}') for i in col
        )