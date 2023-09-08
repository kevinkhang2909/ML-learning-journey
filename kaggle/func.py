import polars as pl


def preprocess(df):
    df = (
        df.with_columns(pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S%z'))
        .sort(['timestamp'], descending=True)
    )
    return df

def print_stats(df):
    print(df['timestamp'].max().date(), df['timestamp'].min().date())
    print(f"# series: {df['series_id'].n_unique()}")

