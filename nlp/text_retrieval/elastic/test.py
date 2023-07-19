from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# Data input
path = Path.home() / 'OneDrive - Seagroup/ai/nlp/cat_tag/clean_all_v2.ftr'
df = pd.read_feather(path).head(1_000_000)
print(df.shape)

# Elasticsearch
es = Elasticsearch("http://localhost:9200")
es.info().body

# check if exist
indexes = 'item'
if es.indices.exists(index=indexes):
    es.indices.delete(index=indexes)

mappings = {
        'properties': {
            'item_name': {'type': 'text'},
            'cat': {'type': 'text', 'analyzer': 'standard'}
    }
}
es.indices.create(index=indexes, mappings=mappings)

bulk_data = []
for i, (text, cat) in enumerate(df[['text_edit', 'all_cat']].values):
    bulk_data.append(
        {
            '_index': indexes,
            '_id': i,
            '_source': {
                'item_name': text,
                'cat': cat
            }
        }
    )
bulk(es, bulk_data)
es.indices.refresh(index=indexes)
print(es.cat.count(index=indexes, format='json'))

# search
for item in tqdm(df[['text_edit']].head(100_000).values[:]):
    resp = es.search(
        size=5,
        index=indexes,
        query={
            'match': {
                'item_name': item[0]
            }
        }
    )
    resp.body
