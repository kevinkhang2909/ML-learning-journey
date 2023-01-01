import numpy as np
import pandas as pd
from re import sub
from time import time
from sparse_dot_topn import awesome_cossim_topn
from sklearn.feature_extraction.text import TfidfVectorizer


class TextMatch:
    """
    Text Matching Wrapper
    """
    def __init__(self, base: list, source: list, top_k: int = 5, similarity_thres: float = .5):
        self.match_sparse_matrix = None
        self.base = base
        self.source = source
        self.top_k = top_k
        self.similarity_thres = similarity_thres

    @staticmethod
    def ngrams_func(string, n=3):
        string = sub(r'[,-./]|\sBD', r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def get_matches_df(self):
        non_zeros = self.match_sparse_matrix.nonzero()
        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]

        nr_matches = sparsecols.size
        left_side = np.empty([nr_matches], dtype=object)
        right_side = np.empty([nr_matches], dtype=object)
        similairity = np.zeros(nr_matches)

        for index in range(0, nr_matches):
            left_side[index] = self.base[sparserows[index]]
            right_side[index] = self.source[sparsecols[index]]
            similairity[index] = self.match_sparse_matrix.data[index]

        return pd.DataFrame({'base': left_side, 'source': right_side, 'similarity': similairity})

    def run_match(self):
        # vectorize
        t1 = time()
        vectorizer = TfidfVectorizer(min_df=1, analyzer=self.ngrams_func)
        maxtrix_base = vectorizer.fit_transform(self.base)
        maxtrix_source = vectorizer.transform(self.source)
        print(f"Process vectorize: {round(time() - t1, 2)}s")

        # match
        t1 = time()
        self.match_sparse_matrix = awesome_cossim_topn(maxtrix_base, maxtrix_source.transpose(),
                                                       self.top_k, self.similarity_thres, use_threads=True, n_jobs=4)
        matches_df = self.get_matches_df()
        matches_df['similarity'] = matches_df['similarity'].map(lambda x: round(x, 2))
        matches_df['rank'] = matches_df.groupby(['base'])['source'].transform(lambda x: pd.factorize(x)[0]).add(1)
        print(f"Process optimized: {round(time() - t1, 2)}s")
        print(f'{len(matches_df.base.unique()):,.0f} items in BASE '
              f'match {len(matches_df.source.unique()):,.0f} items in SOURCE '
              f'with top {self.top_k} match and similarity threshold: {self.similarity_thres}')
        return matches_df
