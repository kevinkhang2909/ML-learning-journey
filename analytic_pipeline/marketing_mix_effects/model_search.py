from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial

from lightgbm import LGBMRegressor, log_evaluation
from optuna import trial, create_study, samplers, logging
from sklearn.metrics import mean_squared_error
import shap

from func import Preprocess, Training

logging.set_verbosity(logging.WARNING)


def train_search(trial, data, features, label, adstock_params, media_features, is_multiobjective):
    model_params = {
        'objective': trial.suggest_categorical('objective', ['poisson', 'gamma']),
        'metric': 'rmse',
        'random_state': 42,
        'n_estimators': trial.suggest_int("n_estimators", 100, 800),
    }
    df = data.copy()
    ads_stock = {i: trial.suggest_float(f"adstock_alpha_{i}", val[0], val[1]) for i, val in adstock_params.items()}
    for feature, val in ads_stock.items():
        df[f'{feature}_adstock'] = Preprocess.adstock_geometric(df[feature], alpha=val)

    # Model: Split & Train
    train = Training(features, label, df)
    x, y, dataset = train.train_test_split('2019-01-01', '2019-06-01')
    model = LGBMRegressor(**model_params)
    model.fit(x['train'], y['train'],
              eval_set=[(x['valid'], y['valid'])],
              callbacks=[log_evaluation(0)])
    explainer = shap.TreeExplainer(model)
    df_shap = pd.DataFrame(explainer.shap_values(x['test']), columns=features)
    effect_share = train.shap_effects(df_shap, media_features)

    scores, rssds = [], []
    rmse = mean_squared_error(y_true=y['test'], y_pred=model.predict(x['test']), squared=False)
    scores.append(rmse)

    if is_multiobjective:
        decomp_rssd = train.rssd(effect_share['contribution'].values, effect_share['distribution'].values)
        rssds.append(decomp_rssd)

    trial.set_user_attr('scores', scores)
    trial.set_user_attr('params', model_params)
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
label = 'revenue'
adstock_features_params = {
    'tv_S': (0.3, 0.8),
    'ooh_S': (0.1, 0.4),
    'print_S': (0.1, 0.4),
    'facebook_S': (0.0, 0.4),
    'search_S': (0.0, 0.3),
    'newsletter': (0.1, 0.4),
}

multi_objective = False
n_trials = 50
if not multi_objective:
    study_mmm = create_study(direction='minimize', sampler=samplers.TPESampler(seed=42))
else:
    study_mmm = create_study(directions=['minimize', 'minimize'], sampler=samplers.NSGAIISampler(seed=42))

opt_func = partial(train_search,
                   data=df,
                   features=features,
                   label=label,
                   media_features=media_channels,
                   adstock_params=adstock_features_params,
                   is_multiobjective=multi_objective)
study_mmm.optimize(opt_func, n_trials=n_trials, show_progress_bar=True)

best_params = study_mmm.best_trial.user_attrs["params"]
best_adstock = study_mmm.best_trial.user_attrs["adstock_alphas"]
