import os
import re
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tarfile
import logging

from datasets import DataSet

logger = logging.getLogger('20news')


class Newsgroups(DataSet):
    LABELS = ['alt.atheism',  # 0
              'comp.graphics',  # 1
              'comp.os.ms-windows.misc',  # 2
              'comp.sys.ibm.pc.hardware',  # 3
              'comp.sys.mac.hardware',  # 4
              'comp.windows.x',  # 5
              'misc.forsale',  # 6
              'rec.autos',  # 7
              'rec.motorcycles',  # 8
              'rec.sport.baseball',  # 9
              'rec.sport.hockey',  # 10
              'sci.crypt',  # 11
              'sci.electronics',  # 12
              'sci.med',  # 13
              'sci.space',  # 14
              'soc.religion.christian',  # 15
              'talk.politics.guns',  # 16
              'talk.politics.mideast',  # 17
              'talk.politics.misc',  # 18
              'talk.religion.misc']  # 19

    MIN_DF = 50
    MAX_DF = 0.35
    MAX_VOCAB = None
    MIN_TOKENS = 10

    def __init__(self, raw_data_dir, temp_dir):
        super().__init__(raw_data_dir)
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        self.data_file = f'{self.temp_dir}/data.npy'
        self.vecs = None
        self.labels = None

    def load(self):
        if os.path.isfile(self.data_file):
            with open(self.data_file, 'rb') as f:
                self.labels = np.load(f)
                self.vecs = np.load(f)
        else:
            labels, vecs = self._prepare_raw()
            self.vecs = vecs
            self.labels = labels
            with open(self.data_file, 'wb') as f:
                np.save(f, self.labels)
                np.save(f, self.vecs)

    def _process_file(self, tar, member):
        file = tar.extractfile(member).read().decode(errors='replace')
        name = member.name
        label = name.split('/')[1]
        name = name.split('/')[2]
        # first two rows are always subject and author
        file = ' '.join(file.split('\n')[3:])
        # remove anything that is not text
        file = re.sub(r'[^a-zA-Z.,:;!? ]', ' ', file)
        return label, name, file

    def _load_raw(self):
        with tarfile.open(f'{self.raw_data_dir}/20news-18828.tar.gz', 'r:gz') as f:
            logger.debug('No prepared temporary data file, loading raw...')
            data = [self._process_file(f, member) for member in f if member.isfile()]
            logger.debug(f'Data loaded from {len(data)} files.')

            logger.debug('Assigning labels...')
            label_map = {k: v for v, k in enumerate(self.LABELS)}
            labels = [label_map[d[0]] for d in data]
            return labels, data

    def _prepare_raw(self):
        labels, data = self._load_raw()

        logger.debug('Fitting transformer...')
        texts = [d[2] for d in data]

        vectorizer = TfidfVectorizer(max_df=self.MAX_DF, min_df=self.MIN_DF, max_features=self.MAX_VOCAB)
        vecs = vectorizer.fit_transform(texts)
        logger.debug(f'Max DF: {self.MAX_DF}, min DF: {self.MIN_DF}, vocab limiter: {self.MAX_VOCAB}')
        logger.debug(f'Vocab size: {len(vectorizer.vocabulary_)}; '
                     f'first values: {list(vectorizer.vocabulary_)[:1000]}')
        logger.debug(f'Dropped tokens: {len(vectorizer.stop_words_)};'
                     f' examples: {list(vectorizer.stop_words_)[:1000]}')

        doc_filter = np.asarray(np.sum(vecs > 0, axis=1) > self.MIN_TOKENS)
        doc_filter = doc_filter.reshape((doc_filter.shape[0],))
        logger.debug(f'Docs with <{self.MIN_TOKENS} tokens: {vecs.shape[0] - sum(doc_filter)}/{vecs.shape[0]}')

        return np.array(labels)[doc_filter], vecs[doc_filter].toarray()

    def run_parameter_sweep(self, min_dfs=None, max_dfs=None, min_count=10):
        labels, data = self._load_raw()
        texts = [d[2] for d in data]
        if min_dfs is None:
            min_dfs = [5, 10, 100, 500]
        if max_dfs is None:
            max_dfs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        for min_df in min_dfs:
            for max_df in max_dfs:
                vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
                vecs = vectorizer.fit_transform(texts)
                logger.debug(f'Max DF: {max_df}, min DF: {min_df}')
                logger.debug(f'Vocab size: {len(vectorizer.vocabulary_)}; ')
                logger.debug(f'Dropped tokens: {len(vectorizer.stop_words_)}; ')
                logger.debug(f'Docs with <{min_count} tokens: {sum(np.sum(vecs > 0, axis=1) < min_count)}')

    def __len__(self) -> int:
        return self.labels.shape[0]

    @property
    def data_labels(self) -> List[Union[int, str]]:
        return self.LABELS

    def generate_data(self):
        for i in self.get_data():
            yield i

    def get_data(self):
        return self.vecs

    def generate_labels(self):
        for i in self.get_labels():
            yield i

    def get_labels(self):
        return self.labels
