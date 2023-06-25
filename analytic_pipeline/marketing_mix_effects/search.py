from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial

import shap
from optuna import create_study, samplers
from lightgbm import LGBMRegressor, log_evaluation
from sklearn.metrics import mean_squared_error

from func import Training, Preprocess


def train(data, adstock_params, model_params, all_features, label_name, media_features):
    data_tmp = data.copy()
    for feature, val in adstock_params.items():
        data_tmp[f'{feature}_adstock'] = Preprocess.adstock_geometric(data_tmp[feature], alpha=val)

    # Model: Split & Train
    train = Training(all_features, label_name, data_tmp)
    X, y, dataset = train.train_split('2019-01-01', '2019-06-01')
    model = LGBMRegressor(**model_params)
    model.fit(X['train'], y['train'],
              eval_set=[(X['valid'], y['valid'])],
              callbacks=[log_evaluation(0)])
    explainer = shap.TreeExplainer(model)
    df_shap = pd.DataFrame(explainer.shap_values(X['test']), columns=all_features)
    spend_effect_share = train.shap_effects(df_shap, media_features)
    return model, X, y, dataset, spend_effect_share


def optuna_trial(trial, data,
                 media_features, adstock_features_params, all_features, label_name,
                 is_multiobjective: bool = False):
    # Params
    params = {
        'objective': trial.suggest_categorical('objective', ['poisson', 'gamma']),
        'metric': trial.suggest_categorical('metric', ['mae', 'mape', 'rmse']),
        'random_state': 42,
        'n_estimators': trial.suggest_int("n_estimators", 100, 8000),
        # "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    scores = []
    rssds = []
    # Adstock
    ads_stock = {i: trial.suggest_float(f"adstock_alpha_{i}", val[0], val[1])
                 for i, val in adstock_features_params.items()}
    model, X, y, dataset, spend_effect_share = train(data, ads_stock, params, all_features, label_name, media_features)
    pred = model.predict(X['test'])
    rmse = mean_squared_error(y_true=y['test'], y_pred=pred, squared=False)
    scores.append(rmse)

    if is_multiobjective:
        decomp_rssd = train.rssd(spend_effect_share['contribution'].values,
                                 spend_effect_share['distribution'].values)
        rssds.append(decomp_rssd)

    trial.set_user_attr('scores', scores)
    trial.set_user_attr('params', params)
    trial.set_user_attr('adstock_alphas', ads_stock)
    trial.set_user_attr('rssds', rssds)

    if not is_multiobjective:
        return np.mean(scores)
    return np.mean(scores), np.mean(rssds)


# Data: Ads
path = Path.home() / 'OneDrive - Seagroup/mkt/mkt_mix'
df = pd.read_csv(path / 'ad_fb.csv', parse_dates=['DATE'])
print(df.shape)

organic_channels = ['newsletter']
media_channels = ['tv_S', 'ooh_S', 'print_S', 'facebook_S', 'search_S']
features = ['competitor_sales_B'] + media_channels + organic_channels
target = 'revenue'
adstock_features_params = {
    'tv_S': (0.3, 0.8),
    'ooh_S': (0.1, 0.4),
    'print_S': (0.1, 0.4),
    'facebook_S': (0.0, 0.4),
    'search_S': (0.0, 0.3),
    'newsletter': (0.1, 0.4),
}

multi_objective = False
if not multi_objective:
    study_mmm = create_study(direction='minimize',
                             sampler=samplers.TPESampler(seed=42))
else:
    study_mmm = create_study(directions=['minimize', 'minimize'],
                             sampler=samplers.NSGAIISampler(seed=42))

opt_func = partial(optuna_trial,
                   data=df,
                   media_features=media_channels,
                   all_features=features,
                   label_name=target,
                   adstock_features_params=adstock_features_params,
                   is_multiobjective=multi_objective)
study_mmm.optimize(opt_func, n_trials=200, show_progress_bar=True)

best_params = study_mmm.best_trial.user_attrs["params"]
best_adstock = study_mmm.best_trial.user_attrs["adstock_alphas"]
model, X, y, dataset, spend_effect_share = train(df, best_adstock, best_params, features, target, media_channels)
