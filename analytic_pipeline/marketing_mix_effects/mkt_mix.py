from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from func import AdstockGeometric, Report, Data
import optuna as opt
import shap
from lightgbm import LGBMRegressor


# Data: Ads
path = Path.home() / 'OneDrive - Seagroup/mkt/mkt_mix'
df = pd.read_csv(path / 'ad_fb.csv', parse_dates=['DATE'])
print(df.shape)

# Data: Holidays
data_input = Data(path)
df_holiday = data_input.holiday('holiday.csv')

# Data: Prophet
prophet_data = df.rename(columns={'revenue': 'y', 'DATE': 'ds'})
prophet_data = pd.concat([prophet_data, pd.get_dummies(prophet_data["events"], drop_first=True, prefix="events")], axis=1)
prophet_predict = Data.seasonal_prophet(df_holiday, prophet_data)

prophet_columns = [col for col in prophet_predict.columns if not col.endswith("upper") and not col.endswith("lower")]
events_numeric = prophet_predict[prophet_columns].filter(like="events_").sum(axis=1)

final_data = df.copy()
for i in ['trend', 'yearly', 'holidays']:
    final_data[i] = prophet_predict[i]
final_data["events"] = (events_numeric - np.min(events_numeric)).values

# Features:
media_channels = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
organic_channels = ["newsletter"]
features = ["trend", "season", "holiday", "competitor_sales_B", "events"] + media_channels + organic_channels
adstock_features = media_channels + organic_channels
adstock_features_params = {
    'tv_S_adstock': (0.3, 0.8),
    'ooh_S_adstock': (0.1, 0.4),
    'print_S_adstock': (0.1, 0.4),
    'facebook_S_adstock': (0.0, 0.4),
    'search_S_adstock': (0.0, 0.3),
    'newsletter_adstock': (0.1, 0.4),
}
target = "revenue"

# Model:
for feature in adstock_features:
    adstock_param = f"{feature}_adstock"
    min_, max_ = adstock_features_params[adstock_param]
    x_feature = final_data[feature].values.reshape(-1, 1)
    temp_adstock = AdstockGeometric(alpha=.5).fit_transform(x_feature)
    final_data[feature] = temp_adstock

x_train = final_data.query('DATE <= "2019-06-01"')[features]
y_train = final_data.query('DATE <= "2019-06-01"')[target].values

x_test = final_data.query('DATE >= "2019-06-01"')[features]
y_test = final_data.query('DATE >= "2019-06-01"')[target].values

model = LGBMRegressor(random_state=0)
model.fit(x_train, y_train)

pred = model.predict(x_test)
rmse = mean_squared_error(y_true=y_test, y_pred=pred, squared=False)

# shap explainer
explainer = shap.TreeExplainer(model)
df_shap_values = pd.DataFrame(explainer.shap_values(x_test), columns=features)

spend_effect_share = Report.effects(df_shap_values=df_shap_values, media_channels=media_channels, df_original=x_test)
decomp_rssd = Report.rssd(effect_share=spend_effect_share.effect_share.values, spend_share=spend_effect_share.spend_share.values)
print(f"DECOMP.RSSD: {decomp_rssd}")

# print(plot_spend_vs_effect_share(spend_effect_share, figure_size = (15, 7)))
