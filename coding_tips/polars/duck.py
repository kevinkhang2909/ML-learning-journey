from pathlib import Path
import duckdb


file_path = str(Path.home() / 'Downloads/2021_Yellow_Taxi_Trip_Data.csv')
# duckdb.read_csv(file_path)
conn = duckdb.connect()
result = conn.execute(f"""
SELECT *
FROM read_csv_auto(file_path);
""")

# # Pandas
# result.df()

# # Arrow
# result.arrow()
#
# # Polars
# import polars as pl
# pl.from_arrow(result.arrow())