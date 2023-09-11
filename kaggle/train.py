from pathlib import Path
import polars as pl
from sklearn.model_selection import train_test_split
from func import Extract, Training
import lightgbm as lgb


path = Path.home() / 'OneDrive - Seagroup/ai/kaggle_dataset/child-mind-institute-detect-sleep-states'
col = ['timestamp']
window = 7
df = (
    pl.read_parquet(path / 'clean.parquet', n_rows=100_000)
    .pipe(Extract.month_day, col=col[0])
    # .pipe(Extract.cycle_time)
    # .pipe(Extract.trend, col=col, window=window)
    # .pipe(Extract.season, col=col, window=window)
    # .pipe(Extract.lag, col=col, window=window)
    .drop_nulls()
)
print(df.shape, df.schema.keys())

# train
col_sum, names = ['total_order', 'gmv_usd'], ['ado', 'adg']
label_name = 'is_wakeup'
drop_col = ['timestamp', 'series_id', 'grass_date', 'grass_hour'] + col + [label_name]
model_path = path / f'models'
all_features = list(filter(lambda i: i not in drop_col, list(df.schema.keys())))

X = df[all_features]
y = df[label_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
train = lgb.Dataset(X_train.to_numpy(), y_train.to_numpy())
valid = lgb.Dataset(X_test.to_numpy(), y_test.to_numpy())

param = {
        'objective': 'binary',
        'metric': 'auc',
        'seed': 42,
        'verbose': -1,
        'feature_pre_filter': False,
        'deterministic': True,
}
model = lgb.train(param, train,
                  valid_sets=[valid],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
preds = model.predict(X_test.to_numpy(), num_iteration=model.best_iteration)
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(preds, y_test)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, preds))
# train = Training(all_features, label_na/o_pandas()

#
#     # re train
#     if re_train:= True:
#         f = partial(objective, lgb_data=lgb_data, all_features=all_features, label_name=label_name, dataset=dataset)
#         study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction='minimize')
#         study.optimize(f, n_trials=500, show_progress_bar=True)
#         print(f'Best trial: {study.best_trial.value:,.0f}')
#
#         # checkpoints
#         model_files = [*model_path.glob(f'*{label_name}*.pkl')]
#         model_best_path = model_path / f'{label_name}_{study.best_trial.number}.pkl'
#         model_best_path.rename(model_path / f'best_{label_name}_{study.best_trial.number}_{study.best_trial.value}.pkl')
#         for i in model_files:
#             if i != model_best_path:
#                 i.unlink()
#
#     model_best_path = str([*model_path.glob(f'*{label_name}*.pkl')][0])
#
#     # reload
#     model = lgb.Booster(model_file=model_best_path)
