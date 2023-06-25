import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

plt.style.use('ggplot')


class Training:
    def __init__(self, features, label, data):
        self.features = features
        self.label = label
        self.data = data
        self.X = None
        self.y = None
        self.dataset = None
        self.col_metric = ['predicted', 'true_label']

    def train_test_split(self, date_from, date_to):
        self.dataset = {
            'train': self.data.query(f'DATE < "{date_from}"').reset_index(drop=True),
            'valid': self.data.query(f'"{date_from}" <= DATE < "{date_to}"').reset_index(drop=True),
            'test': self.data.query(f'DATE >= "{date_to}"').reset_index(drop=True),
        }
        self.X = {i: data[self.features].fillna(-1).to_numpy() for i, data in self.dataset.items()}
        self.y = {i: data[self.label].to_numpy() for i, data in self.dataset.items()}
        return self.X, self.y, self.dataset

    def results(self, model, col_key):
        results = {}
        for name in self.dataset:
            pred = model.predict(self.X[name]).astype(int)
            # results
            tmp = self.dataset[name][col_key].copy()
            tmp['predicted'] = pred
            tmp['true_label'] = self.y[name]

            tmp['percentage'] = tmp['predicted'] / tmp['true_label'] - 1
            tmp['data_type'] = name
            results[name] = tmp
        return results

    def feature_importance(self, model):
        zip_ = zip(self.all_features, model.feature_importances_)
        col = 'numbers of times the feature is used'
        feature_important = pd.DataFrame(zip_, columns=['feature', col])
        feature_important.sort_values(col, ascending=False, inplace=True)
        return feature_important

    def plot(self, data):
        fig, axes = plt.subplots(1, len(data), figsize=(25, 6))
        axes = axes.flatten()
        for idx, (name, val) in enumerate(data.items()):
            tmp = val.groupby('DATE')[self.col_metric].sum()
            tmp.plot(kind='line', ax=axes[idx])
            mae_val = mean_absolute_error(tmp['true_label'], tmp['predicted'])
            print(mae_val)
            axes[idx].set_title(f'{name} - MAE: {mae_val:,.0f}')
        fig.tight_layout()

    def shap_effects(self, df_shap, target_channels):
        # Effects:
        effect = pd.DataFrame(df_shap[target_channels].abs().sum(), columns=['contribution'])
        effect_pct = effect / effect.sum()

        # Shares:
        tmp = self.dataset['test'][target_channels].sum() / self.dataset['test'][target_channels].sum().sum()
        spends_pct = pd.DataFrame(tmp, columns=['distribution'])

        # Merge
        final = pd.merge(effect_pct, spends_pct, left_index=True, right_index=True)
        final = final.reset_index().rename(columns={'index': 'test_set_channels'})
        return final

    def plot_each_channels(self, df_shap, target_channels, name, path):
        for channel in target_channels:
            mean_spend = self.dataset['test'].loc[self.dataset['test'][channel] > 0, channel].mean()
            fig, ax = plt.subplots(figsize=(15, 6))
            # plot
            sns.regplot(x=self.dataset['test'][channel], y=df_shap[channel],
                        scatter_kws={'alpha': 0.65}, lowess=True, ax=ax, label=channel)
            ax.set_title(f'{channel}: {name} vs Shapley')
            ax.axhline(0, linestyle="--", alpha=0.5)
            ax.axvline(mean_spend, linestyle="--", color="red", alpha=0.5,
                       label=f"Average {name}: {int(mean_spend):,.0f}")
            ax.set_xlabel(f"{channel} {name}")
            ax.set_ylabel(f'SHAP Value for {channel}')
            fig.legend()
            fig.tight_layout()
            fig.savefig(path / f'{channel}.png', dpi=300)

    @staticmethod
    def rssd(effect_share, spend_share):
        """
        Root-sum-square distance, a major innovation of Robyn
        eliminates the majority of "bad models"
        (larger prediction error and/or unrealistic media effect like the smallest channel getting the most effect
        """
        return np.sqrt(np.sum((effect_share - spend_share) ** 2))


class Preprocess:
    @staticmethod
    def time_extract(target, data):
        data[f'{target}_year'] = data[target].dt.year
        data[f'{target}_month'] = data[target].dt.month
        data[f'{target}_date'] = data[target].dt.day
        return data

    @staticmethod
    def time_cycle(target, data, max_val):
        data[f'{target}_sin'] = np.sin(2 * np.pi * data[target] / max_val)
        data[f'{target}_cos'] = np.cos(2 * np.pi * data[target] / max_val)
        return data

    @staticmethod
    def time_feature(target, df):
        df = Preprocess.time_extract(target, df)
        df[f'days_dif_spike'] = df[f'{target}_month'] - df[f'{target}_date']
        return df

    @staticmethod
    def seasonal(data, num_day, target_col):
        for name in target_col:
            col_trend, col_season = f'trend_r{num_day}d_{name}', f'season_r{num_day}d_{name}'
            data[col_trend] = data[name].rolling(num_day).mean()
            data.eval(f'{col_season } = {name} - {col_trend}', inplace=True)
        data.bfill(inplace=True)
        return data

    @staticmethod
    def adstock_geometric(x, alpha):
        x_decayed = np.zeros_like(x)
        x_decayed[0] = x[0]

        for xi in range(1, len(x_decayed)):
            x_decayed[xi] = x[xi] + alpha * x_decayed[xi - 1]
        return x_decayed
