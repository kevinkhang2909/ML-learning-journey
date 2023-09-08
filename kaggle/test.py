from pathlib import Path
import polars as pl
from func import print_stats, preprocess


path = Path.home() / 'OneDrive - Seagroup/ai/kaggle_dataset/child-mind-institute-detect-sleep-states'

# csv
print('csv')
files = [*path.glob('*.csv')]
df = (
    pl.read_csv(files[0])
    .pipe(preprocess)
    # .to_pandas()
)
print_stats(df)
print(f"# nights: {df['night'].n_unique()}")
print(f"events: {df['event'].value_counts()}")

# parquet
print('parquet')
files = next(path.glob('train*.parquet'))
df_full = (
    pl.read_parquet(files, n_rows=100_000)
    .pipe(preprocess)
    # .to_pandas()
)
# print_stats(df_full)
# check nulls
# null_col = df.select(pl.all().is_null().sum()).to_dicts()[0]
# print(null_col)
#
# df_night = df.groupby('night').agg(pl.col('series_id').count()).to_pandas()
# df_series = df.groupby('series_id').agg(pl.col('event').count()).to_pandas()
# df.filter(pl.col('night') == 1)