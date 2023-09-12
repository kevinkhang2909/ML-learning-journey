from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from func import Extract
import lightgbm as lgb
from itertools import groupby
from sklearn.metrics import accuracy_score, classification_report


path = Path.home() / 'OneDrive - Seagroup/ai/kaggle_dataset/child-mind-institute-detect-sleep-states'
col = ['timestamp']
window = 7
df = (
    pl.read_parquet(path / 'clean.parquet')
    .pipe(Extract.month_day, col=col[0])
    .pipe(Extract.cycle_time)
    # .pipe(Extract.trend, col=col, window=window)
    # .pipe(Extract.season, col=col, window=window)
    # .pipe(Extract.lag, col=col, window=window)
    .drop_nulls()
)
print(df.shape, df.schema.keys())

# train
label_name = 'is_wakeup'
drop_col = ['timestamp', 'series_id', 'grass_date', 'grass_hour'] + col + [label_name]
model_path = path / f'models'
all_features = list(filter(lambda i: i not in drop_col, list(df.schema.keys())))

X_train, X_test, y_train, y_test = train_test_split(df, df[label_name], test_size=0.33, random_state=42)
train = lgb.Dataset(X_train[all_features].to_numpy(), y_train.to_numpy())
valid = lgb.Dataset(X_test[all_features].to_numpy(), y_test.to_numpy())
del df

param = {
        'objective': 'binary',
        'metric': 'auc',
        'seed': 42,
        'verbose': 1,
        'feature_pre_filter': False,
        'force_col_wise': True,
        'deterministic': True,
}
model = lgb.train(param, train,
                  valid_sets=[valid],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
preds = model.predict(X_test[all_features].to_numpy(), num_iteration=model.best_iteration)

test = X_test[['series_id']].to_pandas().copy()
test['score'] = np.where(preds > 0.5, 1, 0)
test['label'] = y_test

accuracy = accuracy_score(test['score'], test['label'])
print(f'Accuracy: {accuracy}')
print(classification_report(test['label'], test['score']))

zip_ = zip(all_features, model.feature_importance())
feature_importance = (
    pl.DataFrame(zip_, schema=['feature', '# times the feature is used'])
    .sort('# times the feature is used', descending=True)
    .to_pandas()
)
print(feature_importance)

# baseline
# Accuracy: 0.9034865806061912
#               precision    recall  f1-score   support
#            0       0.79      0.81      0.80   9853570
#            1       0.94      0.93      0.94  31336070
#     accuracy                           0.90  41189640
#    macro avg       0.87      0.87      0.87  41189640
# weighted avg       0.90      0.90      0.90  41189640

