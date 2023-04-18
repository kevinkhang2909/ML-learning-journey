import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder


def preprocess(file_name, path):
    # input
    data = pd.read_table(path / file_name, engine='pyarrow')

    # clean
    # data = data[data['price'] >= 1.0].reset_index(drop=True)
    data['desc_len'] = data['item_description'].str.split(' ').str.len()
    data['desc_len'] = np.where(data['item_description'] == 'No description yet', 0, data['desc_len'])
    data['name_len'] = data['name'].str.split(' ').str.len()

    for i in data.select_dtypes(include='object').columns:
        data[i] = data[i].str.lower()

    # missing values
    fillna = 'missing/missing/missing'
    col_cat = ['subcategory_1', 'subcategory_2', 'subcategory_3']
    tmp = data['category_name'].fillna(fillna).replace('', fillna).str.split('/').str[:3].values.tolist()
    tmp = pd.DataFrame(tmp, columns=col_cat)
    tmp[col_cat] = oe.fit_transform(tmp[col_cat])
    data = pd.concat([data, tmp], axis=1)

    data['brand_name'] = data['brand_name'].fillna('missing').replace('', 'missing')

    # text
    data['text'] = data['name'] + ' . ' + data['brand_name'] + ' . ' + data['item_description']

    # drop
    drop_col = ['brand_name', 'name', 'category_name', 'item_description']
    data.drop(columns=drop_col, inplace=True)

    # export
    data.to_feather(path / f"{file_name.split('.')[0]}.ftr")
    return data


oe = OrdinalEncoder()
path = Path.home() / 'OneDrive - Seagroup/ai/kaggle_dataset/mercari-price-suggestion-challenge'
train = preprocess('train.tsv', path)
# test = preprocess('test.tsv', path)
