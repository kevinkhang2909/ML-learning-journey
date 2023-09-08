import polars as pl


class Preprocess:
    @staticmethod
    def format_time(df):
        df = (
            df.with_columns(pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S%z'))
            .sort(['timestamp'], descending=True)
        )
        return df

    @staticmethod
    def check_null(df):
        return df.select(pl.all().is_null().sum()).to_dicts()[0]


def print_stats(df):
    print(df.shape)
    print(df['timestamp'].max().date(), df['timestamp'].min().date())
    print(f"# series: {df['series_id'].n_unique()}")

