import numpy as np
import pandas as pd
from re import sub
from time import time
from sparse_dot_topn import awesome_cossim_topn
from sklearn.feature_extraction.text import TfidfVectorizer


def ngrams_func(string, n=3):
    string = sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def get_matches_df(sparse_matrix, base, source):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = base[sparserows[index]]
        right_side[index] = source[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'base': left_side, 'source': right_side, 'similarity': similairity})


def run_match(base, source, top, similarity):
    t1 = time()
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams_func)
    maxtrix_base = vectorizer.fit_transform(base)
    maxtrix_source = vectorizer.transform(source)
    print(f"process vectorize: {round(time() - t1, 2)}s")

    # match
    t1 = time()
    matches = awesome_cossim_topn(maxtrix_base, maxtrix_source.transpose(), top, similarity, use_threads=True, n_jobs=4)
    matches_df = get_matches_df(matches, base=base, source=source)
    matches_df['similarity'] = matches_df['similarity'].map(lambda x: round(x, 2))
    print(f"process optimized: {round(time() - t1, 2)}s")
    print(f'{len(matches_df.base.unique()):,.0f} skus in BASE match {len(matches_df.source.unique()):,.0f} skus in SOURCE')
    return matches_df
