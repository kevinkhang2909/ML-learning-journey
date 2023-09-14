from pathlib import Path
import duckdb
from tqdm import tqdm
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from func import check_null, eda_series


plt.style.use('ggplot')
matplotlib.use('Agg')
path = Path.home() / 'OneDrive - Seagroup/ai/kaggle_dataset/child-mind-institute-detect-sleep-states'

# csv
file_csv = next(path.glob('train*.csv'))
query = f"""select series_id
, night
, event
, cast(step as int) step
, cast(timestamp as varchar) as timestamp
, case 
    when event = 'wakeup' then 1
    when event = 'onset' then 0
    else null end is_wakeup
from read_csv_auto('{file_csv}')"""
df_csv = (
    duckdb.sql(query).pl()
    .with_columns(pl.col('timestamp').str.to_datetime(format='%Y-%m-%d %H:%M:%S'))
    .shrink_to_fit()
)
df_csv = check_null(df_csv)

# parquet
file_parquet = next(path.glob('train*.parquet'))
query = f"""select series_id 
, cast(step as int) step
, anglez
, enmo
, cast(timestamp as varchar) as timestamp
from read_parquet('{file_parquet}')"""
df_parquet = (
    duckdb.sql(query).pl()
    .with_columns(pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S%z').dt.replace_time_zone(None))
    .shrink_to_fit()
)
df_parquet = check_null(df_parquet)

# merge
query = f"""
select p.*
, c.night
, c.is_wakeup
from df_parquet p
left join df_csv c on p.series_id = c.series_id and p.timestamp = c.timestamp
order by p.series_id, p.timestamp
"""
df_merge_full = (
    duckdb.sql(query).pl()
    .with_columns(pl.col(i).forward_fill() for i in ['is_wakeup', 'night'])
    .shrink_to_fit()
)

# filter series_id have 1 labels
tmp = df_merge_full.groupby('series_id').agg(pl.col('is_wakeup').n_unique())
exclude_lst = tmp.filter(pl.col('is_wakeup') < 2)['series_id'].unique().to_list()
print(len(exclude_lst))

df_merge_full = df_merge_full.filter(~pl.col('series_id').is_in(exclude_lst))
print(df_merge_full.shape, df_merge_full['series_id'].n_unique())

# export
df_merge_full.write_parquet(path / 'clean.parquet', use_pyarrow=True)

# plot series
run = df_merge_full['series_id'].unique().to_list()
f = partial(eda_series,  df_merge_full=df_merge_full, df_csv=df_csv, path=path)
with tqdm(total=len(run)) as pbar:
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(f, arg): arg for arg in run}
        results = {}
        for future in as_completed(futures):
            arg = futures[future]
            results[arg] = future.result()
            pbar.update(1)
