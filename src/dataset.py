import os, sys
from os.path import exists
import glob
import random

import numpy as np
import pandas as pd

import torch
import datasets

# NusaWrites Data Loading
def load_dataset(dataset, task, lang, num_sample:int=-1, base_path='./data'):
    data_files = {}
    for path in glob.glob(f'{base_path}/{dataset}-{task}-{lang}-*.csv'):
        split = path.split('-')[-1][:-4]
        data_files[split] = path
        #add path arguments to enable sampled data collection
    output_dataset = datasets.load_dataset('csv', data_files=data_files)
    if num_sample == -1:
        return output_dataset
    else:
        return output_dataset.filter(lambda _, idx: idx < num_sample, with_indices=True)

def load_nlg_tasks():
    meta = []
    for path in glob.glob('./data/nusa_kalimat-mt-*.csv'):
        meta.append(tuple(path.split('/')[-1][:-4].split('-')[:3]))
    meta = sorted(list(set(filter(lambda x: x[1] == 'mt', meta))))
    return { (dataset, task, lang) : load_dataset(dataset, task, lang) for (dataset, task, lang) in meta } 

def load_nlu_tasks():
    meta = []
    for path in glob.glob('./data/nusa_kalimat-emot-*.csv'):
        meta.append(tuple(path.split('/')[-1][:-4].split('-')[:3]))
    return { (dataset, task, lang) : load_dataset(dataset, task, lang) for (dataset, task, lang) in meta } 

# MasakhaNews Data Loading
def load_masakhanews_dataset(base_path='./masakhanews'):
    dsets = {}
    for path in glob.glob(f'{base_path}/*'):
        lang = path.split('/')[-1]
        dsets[lang] = datasets.DatasetDict({
            'train': datasets.Dataset.from_pandas(pd.read_csv(f'{path}/train.tsv', sep='\t')), 
            'dev': datasets.Dataset.from_pandas(pd.read_csv(f'{path}/dev.tsv', sep='\t')), 
            'test': datasets.Dataset.from_pandas(pd.read_csv(f'{path}/test.tsv', sep='\t'))
        })
    return dsets

def load_bible_data():
    return pd.read_csv('bible_cache/bible_all.csv')

def get_parallel_from_bible(bible_df, src_lang, tgt_lang):
    return bible_df.loc[:,[src_lang, tgt_lang]].applymap(
        lambda x: np.nan if pd.isna(x) else x.strip()
    ).dropna().rename({src_lang: 'text',tgt_lang: 'mt_text'}, axis='columns')

