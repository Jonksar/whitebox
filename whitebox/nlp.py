"""
  --------------------------------------------------
  File Name : nlp.py
  Creation Date : 18-05-2018
  Last Modified : 2019-12-27 Fri 10:58 pm
  Created By : Joonatan Samuel
  --------------------------------------------------
"""
from __future__ import print_function

import gensim
import wget
import string
import zipfile
import os, sys
import numpy as np
import pandas as pd
import re

from annoy import AnnoyIndex

from os.path import expanduser, join
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from nltk.tokenize import sent_tokenize
from enum import Enum
from .skipthoughts import skipthoughts

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.base import TransformerMixin
from nltk.tokenize import sent_tokenize
from enum import Enum
from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from .skipthoughts import skipthoughts
import re
from .utils import first

class SummarizationLengthStrategy(Enum):
    EXPONENTIAL = 1
    LINEAR = 2


class ExtractiveSummarization:
    def __init__(self):
        self._download_pretrained()

        # You would need to download pre-trained models first
        self.model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(self.model)

    def _download_pretrained(self):
        skipthoughts.download_pretrained_skipthoughs()

    def preprocess_clean(self, text):
        # Returns text with all the filtering necessary
        text = re.sub(r'[[0-9]]', ' ', text)
        text = re.sub(r'\n','',text)
        text = re.sub(r'\xa0', ' ', text)
        text = re.sub(r'[()[\]{}]',' ',text)
        text = re.sub(r'\s+',' ',text)
        return text


    def summarize(self, text, language="english", amount=0.5, length_strategy=SummarizationLengthStrategy.EXPONENTIAL):
        assert 0. < amount < 1., "Amount should be between 0 and 1"

        # Clean odd characters
        text = self.preprocess_clean(text)

        # Get the sentences
        sentences = sent_tokenize(text, language=language)
        # Find vectors
        encoded = self.encoder.encode(sentences)

        # Finding amount of sentences based on the strategy
        if length_strategy == SummarizationLengthStrategy.EXPONENTIAL:
            n_clusters = int(np.ceil(len(encoded) ** amount))
        elif length_strategy == SummarizationLengthStrategy.LINEAR:
            n_clusters = int(np.ceil(len(encoded) * amount))
        else:
            raise AttributeError("Summarization strategy not supported")

        # Train clustering algorithm
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(encoded)

        avg = []
        closest = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers,
                                                       encoded)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        print('Clustering Finished')
        return summary

class Embedding(TransformerMixin):
    def __init__(self):
        w2v = None

        OUT_DIR = expanduser('~/.whitebox')
        ZIP_NAME = expanduser(join(OUT_DIR, 'glove.6B.zip'))

        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)

        if not os.path.isfile(ZIP_NAME):
            print(" Downloading glove 6B word embeddings ... ")
            filename = wget.download("http://nlp.stanford.edu/data/glove.6B.zip", out=OUT_DIR)

        if not os.path.isfile(join(OUT_DIR, "glove.6B.50d.txt")):
            print(" Extracting word embeddings ... ")
            zip_ref = zipfile.ZipFile(ZIP_NAME, 'r')
            zip_ref.extractall(OUT_DIR)
            zip_ref.close()

        print("Caching word2vec in memory ...")
        with open(join(OUT_DIR, "glove.6B.50d.txt"), "rb") as lines:
            w2v = {line.split()[0].decode('utf8'): list(map(float, line.split()[1:]))
                   for line in lines}

        self.word2vec = w2v

        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(first(self.word2vec.values()))
        self._words = list(w2v.keys())

        if os.path.isfile(OUT_DIR + "/word2vec_reverse_search.ann"):
            self.index = AnnoyIndex(self.dim, metric='angular')
            self.index.load(OUT_DIR + "/word2vec_reverse_search.ann")
        else:
            print("Building reverse search index...")
            self.index = AnnoyIndex(self.dim, metric='angular')  # Length of item vector that will be indexed

            for i, word in enumerate(self._words[:-10]):
                self.index.add_item(i, self.word2vec[word])

            self.index.build(10)  # 10 trees
            self.index.save(OUT_DIR + "/word2vec_reverse_search.ann")

    def fit(self, X, y):
        return self

    def transform(self, X, return_dataframe=False):
        result = []

        for n_sentence, sentence in enumerate(X):
            sentence = sentence.replace('\n', ' ')
            sentence = re.sub( '[%s()[]{}<>]' % string.punctuation, '', sentence).lower().split(' ')
            n_word = 0
            for word in sentence:
                if word == '':
                    continue

                if word in self.word2vec:
                    result.append([n_sentence, n_word] + self.word2vec[word])
                    n_word += 1
                else:
                    print('failed to encode word: \'%s\'' % word)
                    result.append([n_sentence, n_word] + [0 for _ in range(self.dim)])

        if return_dataframe:
            result = pd.DataFrame(result, columns=['sentence_id', 'word_id'] + ['x' + str(i) for i in range(self.dim)])

    def inverse_transform(self, X):
        return [r[2:] for r in result]
