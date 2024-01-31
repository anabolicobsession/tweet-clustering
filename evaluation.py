import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

from constants import EVALUATION_PATH
from utils import format_dict, extract_name, filename_append


METRICS = [
    'ari',
    'ami',
    
    'acc',
    'f1',
    'rec',
    'pre',
    'mrec',
    'mpre',
    
    'ss',
    'vm',
    'h',
    'c',
]
        
def compute_metrics(y, y_pred, X=None):
    h, c, vm = metrics.homogeneity_completeness_v_measure(y, y_pred)
    pre, rec, f1, _ = metrics.precision_recall_fscore_support(y, y_pred)
    return (
        metrics.adjusted_rand_score(y, y_pred),
        metrics.adjusted_mutual_info_score(y, y_pred),
        
        metrics.accuracy_score(y, y_pred, normalize=True),
        np.mean(f1),
        np.mean(rec),
        np.mean(pre),
        np.min(rec),
        np.min(pre),
        
        metrics.silhouette_score(X, y_pred) if X is not None else None,
        vm,
        h,
        c,
    )

def print_metrics(label, metrics_):
    print(label + ': ' + format_dict(dict(zip(METRICS, metrics_))))
    
def metrics_to_df(metrics_, label=None):
    df = pd.DataFrame(columns=METRICS)
    df.loc[label if label is not None else 0] = metrics_
    return df


def match_cluster_labels(c0, c1):
    """Makes the second cluster labels match the first using permutations."""
    mapping = linear_sum_assignment(-metrics.cluster.contingency_matrix(c0, c1))[1].tolist()
    reversed_mapping = [0] * len(mapping)
    for i, v in enumerate(mapping): reversed_mapping[v] = i
    return [reversed_mapping[c] for c in c1]


COLUMNS = [*METRICS, 'time']

class EvaluationResults:
    def __init__(self, name, create_new=False):
        self.df = pd.DataFrame(columns=COLUMNS, dtype='float')
        self.embeddings = {}
        self.clusters = {}
        
        self._df_path = os.path.join(EVALUATION_PATH, name + '.csv')
        self._data_path = os.path.join(EVALUATION_PATH, name + '_data.npz')
        
        if os.path.isfile(self._df_path) and not create_new:
            self.df = pd.read_csv(self._df_path, index_col=0)
            
            for k, v in np.load(self._data_path).items():
                if int(k[-1]) == 0: 
                    self.embeddings[k[:-1]] = v
                else: 
                    self.clusters[k[:-1]] = v
        else:
            Path(EVALUATION_PATH).mkdir(parents=True, exist_ok=True)
    
    def save(self):
        self.df.to_csv(self._df_path)
        np.savez_compressed(self._data_path, **{k + '0': v for k, v in self.embeddings.items()}, **{k + '1': v for k, v in self.clusters.items()})

    def insert(self, key, row, embeddings=None, clusters=None, condition_fun=None, force=False):
        if key not in self.df.index or (condition_fun is not None and condition_fun(row, self.df)) or force:
            self.df.loc[key] = row
            self.df.sort_index(inplace=True)
            self.embeddings[key] = embeddings
            self.clusters[key] = clusters
            self.save()
            return True
        return False

    def drop(self, key=None, condition_fun=None):
        if key is not None:
            self.df.drop(key, inplace=True)
            self.embeddings.pop(key)
            self.clusters.pop(key)
        else:
            keys = self.df.index[self.df.apply(condition_fun, axis=1)]
            self.df.drop(keys, inplace=True)
            self.embeddings = {k:v for k, v in self.embeddings.items() if k not in keys}
            self.clusters = {k:v for k, v in self.clusters.items() if k not in keys}
        self.save()
        
    def rename(self, mapping):
        self.df.rename(mapping, inplace=True)
        self.embeddings = {mapping[k]:v for k, v in self.embeddings.items()}
        self.clusters = {mapping[k]:v for k, v in self.clusters.items()}
        self.save()
