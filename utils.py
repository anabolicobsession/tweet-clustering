import time
from os.path import join

import pandas as pd

from constants import DATA_PATH, DATA_SOURCE_PATH, MODELS_PATH


def format_list(xs, places=2):
    return [f'%.{places}f' % x for x in xs]

def format_dict(d):
    s = ''
    for i, x in enumerate(d.items()): s += f'{x[0]}:' + (f'{x[1]:.2f}' if x[1] is not None else 'NaN') + (', ' if i < len(d) - 1 else '')
    return s

def format_counter(d):
    s = ''
    for i, x in enumerate(d): s += f'{x[0]}:{x[1]}' + (', ' if i < len(d) - 1 else '')
    return s


def extract_name(filename):
    return filename.split('.', maxsplit=1)[0]

def filename_append(filename, s):
    parts = filename.split('.', maxsplit=1)
    return parts[0] + s + '.' + parts[1]


_start_time = None

def start_time():
    global _start_time; _start_time = time.time(); return _start_time
    
def elapsed_time(label=None, verbose=False):
    label = 'time elapsed: ' if label is None else label
    elapsed = time.time() - _start_time
    if verbose or label is not None: print(label + f'{elapsed:.0f}s')
    return elapsed


class Timer:
    def __init__(self):
        self._time = []
        
    def _adjust_size(self, i):
        if len(self._time) < i + 1: self._time.append(None);
        else: self._time = self._time[:i+1]
        
    def start(self, i=None):
        i = i if i is not None else 0
        self._adjust_size(i)
        self._time[i] = time.time()
        return self._time[i]

    def pause(self, i=None):
        i = i if i is not None else 0
        self._time[i] = time.time() - self._time[i]
        return self._time[i]
    
    def set(self, i, elapsed_time):
        self._adjust_size(i)
        self._time[i] = elapsed_time

    def get_total_time(self):
        return sum(self._time)


def save_dataset(dataset, name, **kwargs):
    dataset.to_csv(join(DATA_PATH, name), index=False, **kwargs)

def load_dataset(name, source=False, **kwargs):
    return pd.read_csv(join(DATA_PATH if not source else DATA_SOURCE_PATH, name), **kwargs)

def model_path(name):
    return join(MODELS_PATH, name)
