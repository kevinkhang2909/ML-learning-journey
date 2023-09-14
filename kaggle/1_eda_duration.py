from pathlib import Path
from tqdm import tqdm
import polars as pl
from polars import col
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


plt.style.use('ggplot')
matplotlib.use('Agg')
path = Path.home() / 'OneDrive - Seagroup/ai/kaggle_dataset/child-mind-institute-detect-sleep-states'

# data
df = pl.read_parquet(path / 'clean.parquet')

# plot
df_report_duration = pl.DataFrame()
for series_id in tqdm(df['series_id'].unique()):
    # series_id = '0ce74d6d2106'
    tmp = (
        df.filter(col('series_id') == series_id)
        .with_columns(col('timestamp').shift(1).over(['series_id', 'is_wakeup']).backward_fill().alias('previous_timestamp'))
        .with_columns((col('timestamp') - col('previous_timestamp')).dt.seconds().alias('duration'))
        .with_columns(col('duration').cumsum().over(['series_id', 'is_wakeup']).alias('roll_sum'))
        # .filter(col('step').is_between(4810, 4820))
    )

    # combine labels 0,1 into 1 day
    change = 0
    i = 0
    output = [0]
    for idx, value in enumerate(tmp['is_wakeup'][1:]):
        if value != tmp['is_wakeup'][idx]:
            change += 1
            if change % 2 == 0:
                i += 1
        output.append(i)
    tmp = tmp.with_columns(pl.Series(output).alias('tag'))

    # duration
    duration = (
        tmp.groupby(['series_id', 'tag', 'is_wakeup'], maintain_order=True).agg(
            col('duration').sum(),
            col('timestamp').max().alias('max_timestamp'),
            col('timestamp').min().alias('min_timestamp'),
        )
        .with_columns(
            (col('duration') / 60).alias('duration_min'),
            (col('duration') / 3600).alias('duration_hour')
        )
    )

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    sns.barplot(data=duration.to_pandas(), x='tag', y='duration_min', hue='is_wakeup', ax=ax)
    fig.suptitle(f'series id {series_id}', fontsize=16)
    fig.tight_layout()

    fig.savefig(path / f'media/duration/{series_id}.png', dpi=300)
    plt.close('all')
    plt.close(fig)

    df_report_duration = df_report_duration.vstack(duration)

# export
df_report_duration.write_csv(path / 'report_duration.csv')
