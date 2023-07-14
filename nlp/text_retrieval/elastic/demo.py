from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from pathlib import Path


# Data input
path = Path.home() / 'Downloads/wiki_movie_plots_deduped.csv'
df = (
    pd.read_csv(path)
    .dropna()
    .reset_index()
)
print(df.shape)

# Elasticsearch
es = Elasticsearch("http://localhost:9200")
es.info().body

# check if exist
if es.indices.exists(index='movies'):
    es.indices.delete(index='movies')

mappings = {
    "properties": {
        "title": {"type": "text", "analyzer": "english"},
        "ethnicity": {"type": "text", "analyzer": "standard"},
        "director": {"type": "text", "analyzer": "standard"},
        "cast": {"type": "text", "analyzer": "standard"},
        "genre": {"type": "text", "analyzer": "standard"},
        "plot": {"type": "text", "analyzer": "english"},
        "year": {"type": "integer"},
        "wiki_page": {"type": "keyword"}
    }
}
es.indices.create(index="movies", mappings=mappings)

bulk_data = []
for i, row in df.iterrows():
    bulk_data.append(
        {
            "_index": "movies",
            "_id": i,
            "_source": {
                "title": row["Title"],
                "ethnicity": row["Origin/Ethnicity"],
                "director": row["Director"],
                "cast": row["Cast"],
                "genre": row["Genre"],
                "plot": row["Plot"],
                "year": row["Release Year"],
                "wiki_page": row["Wiki Page"],
            }
        }
    )
bulk(es, bulk_data)
es.indices.refresh(index='movies')
print(es.cat.count(index='movies', format='json'))

# search
resp = es.search(
    index="movies",
    query={
        "bool": {
            "must": {
                "match_phrase": {
                    "cast": "jack nicholson",
                }
            },
            "filter": {"bool": {"must_not": {"match_phrase": {"director": "roman polanski"}}}},
        },
    },
)
resp.body
