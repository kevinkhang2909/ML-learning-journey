from pathlib import Path
import polars as pl
import duckdb
from func import print_stats, Preprocess


path = Path.home() / 'OneDrive - Seagroup/ai/kaggle_dataset/child-mind-institute-detect-sleep-states'

# csv
print('csv')
file = next(path.glob('train*.csv'))
query = f"""select * from read_csv_auto('{file}')"""
df = (
    duckdb.sql(query)
    .pl()
    .drop_nulls()
    # .to_pandas()
)
print_stats(df)
print(f"events: {df['event'].value_counts()}")
print(Preprocess.check_null(df))

query = f"""
with base as 
(
select series_id
, {', '.join([f'{i}(timestamp) {i}_timestamp' for i in ['min', 'max']])}
from df
group by 1
)

select *
, date_diff('day', min_timestamp, max_timestamp) day_diff
from base
"""
df_group = duckdb.query(query)
# parquet
# print('parquet')
# file = next(path.glob('train*.parquet'))
#
# query = f"""select * from read_parquet('{file}')"""
# df = (
#     duckdb.sql(query)
#     .pl()
#     # .pipe(Preprocess.format_time)
# )

# print_stats(df_full)
# print(Preprocess.check_null(df_full))
#
# len(set(df_full['series_id'].unique()) - set(df['series_id'].unique()))
#
# df_train = df_full.join(df, on=['series_id', 'timestamp'], how='left')
# print(df_train.shape)
# print(Preprocess.check_null(df_train))
# print_stats(df_full)
# check nulls


# df_night = df.groupby('night').agg(pl.col('series_id').count()).to_pandas()
# df_series = df.groupby('series_id').agg(pl.col('event').count()).to_pandas()
# df.filter(pl.col('night') == 1)