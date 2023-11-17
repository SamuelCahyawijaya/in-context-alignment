"""nusacrowd zero-shot prompt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ru8DyS2ALWfRdkjOPHj-KNjw6Pfa44Nd
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import csv
from os.path import exists
import glob
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn.functional as F
import datasets

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from dataset_utils import load_dataset
from indexer import DatasetIndexer
from prompter import ICLPrompter, ITCPrompter
from classifier import predict_classification_batch

DEBUG=False

lang_map = {
    'btk': 'Batak', 'sun': 'Sundanese', 'jav': 'Javanese', 
    'mad': 'Madurese', 'mak': 'Buginese', 'min': 'Minangkabau',
    'amh': 'Amharic', 'hau': 'Hausa', 'ibo': 'Igbo', 'lug': 'Luganda', 'pcm': 'Nigerian Pidgin',
    'sna': 'chShona', 'swa': 'Kiswahili', 'xho': 'isiXhosa', 'yor': 'Yorùbá',
    'aym': 'Aymara', 'bzd': 'Bribri', 'cni': 'Asháninka', 'gn': 'Guaraní', 'hch': 'Wixarika',
    'nah': 'Nahuatl', 'oto': 'Otomí', 'quy': 'Quechua', 'shp': 'Shipibo-Konibo', 'tar': 'Rarámuri',
    'ind': 'Indonesian', 'eng': 'English', 'spa': 'Spanish', 'arb': 'Arabic', 'fra': 'French', 
    'deu': 'German', 'hin': 'Hindi', 'ita': 'Italian', 'por': 'Portuguese'
}

dataset_to_metadata_map = {
    # key: (prompt_template, icl_template, iia_template, icl_keys, iia_keys, x_iia_keys)
    'americasnli-spa': (
        'Predice la etiqueta de implicación del siguiente par de oraciones:\n{context}\n{query}',
        'Premisa: "{premise}"; Hipótesis: "{hypothesis}" => {label}',
        '{premise_1} => {premise_2}\n{hypothesis_1} => {hypothesis_2}',
        ['premise', 'hypothesis'], ['premise_1', 'hypothesis_1'], ['premise_2', 'hypothesis_2']
    ),
    'americasnli': (
        'Predict the entailment label of the following pair of sentences:\n{context}\n{query}',
        'Premise: "{premise}"; Hypothesis: "{hypothesis}" => {label}',
        '{premise_1} => {premise_2}\n{hypothesis_1} => {hypothesis_2}',
        ['premise', 'hypothesis'], ['premise_1', 'hypothesis_1'], ['premise_2', 'hypothesis_2']
    ),
    'nusatranslation': (
        'Predict the sentiment label of the following sentence:\n{context}\n{query}',
        '{text} => {label}',
        '{text_1} => {text_2}',
        'text', 'text_1', 'text_2'
    ),
    'masakhanews': (
        'Predict the topic of the following news title:\n{context}\n{query}',
        '{text} => {label}',
        '{text_1} => {text_2}',
        'text', 'text_1', 'text_2'
    ),
    'tweetsentimulti': (
        'Predict the sentiment label of the following tweet:\n{context}\n{query}',
        '{text} => {label}',
        '{text_1} => {text_2}',
        'text', 'text_1', 'text_2'
    ),
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('main_itc_alignment.py <model_path_or_name> <dataset_name> <itc_type> <itc_index_type> <itc_num_exemplar> <label_type> <batch_size>')

    BASE_PATH='./dataset'
    MODEL = sys.argv[1]
    DATASET_NAME = sys.argv[2] # americasnli, nusatranslation, masakhanews
    ITC_TYPE = sys.argv[3] # cross, mono, none
    ITC_INDEX_TYPE = sys.argv[4].split(',') # random, count, tf-idf, sbert
    ITC_EXEMPLAR_COUNT = int(sys.argv[5])
    LABEL_TYPE = sys.argv[6]
    BATCH_SIZE = int(sys.argv[7])
    
    SAVE_NAME = f'itc-{ITC_TYPE}-{"$".join(ITC_INDEX_TYPE)}-{ITC_EXEMPLAR_COUNT}-{LABEL_TYPE}'

    os.makedirs('./metrics_itc', exist_ok=True) 
    os.makedirs('./outputs_itc', exist_ok=True) 

    # Load Dataset
    print('Load Datasets...')
    eval_dsets, icl_dsets, xicl_lang, iia_dsets, itc_dsets, ioa_df = load_dataset(dataset=DATASET_NAME, base_path=BASE_PATH)

    print(f'Loaded {len(eval_dsets)} datasets')
    for i, dset_subset in enumerate(eval_dsets.keys()):
        print(f'{i} {dset_subset}')
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, truncation_side='left')
    if "bloom" in MODEL or "xglm" in MODEL or "gpt2" in MODEL:
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
        tokenizer.pad_token = tokenizer.eos_token # Use EOS to pad label
    # model = torch.compile(model)    
    model.eval()

    metrics = {
        'dataset': [], 'lang': [],
        'accuracy': [], 'macro_f1': [], 'weighted_f1': []
    }
    
    for dset_lang, eval_dset in eval_dsets.items():
        if dset_lang == 'eng':
            continue
        if LABEL_TYPE == 'Target' and dset_lang not in ioa_df.index:
            continue

        print(f'Processing {DATASET_NAME} {dset_lang}')

        ###
        # Preprocessing
        ###

        # Extract Metadata
        prompt_template, icl_template, iia_template, icl_keys, iia_keys, x_iia_keys = dataset_to_metadata_map[DATASET_NAME]
        
        # Prepare Prompter
        query_keys = icl_keys if type(icl_keys) == list else [icl_keys]
        src_keys = x_iia_keys if type(x_iia_keys) == list else [x_iia_keys]
        tgt_keys = iia_keys if type(iia_keys) == list else [iia_keys]
        itc_prompter = ITCPrompter(
            headers = [ # Target Inputs
                f'{lang_map[dset_lang]} {key.split("_")[0].capitalize()}' for key in tgt_keys
            ] + [ # Label
                f'Label'
                # f'{lang_map[dset_lang]} Label' if LABEL_TYPE == 'Target' else f'{lang_map[xicl_lang]} Label'
            ] + ([ # Source Inputs
                f'{lang_map[xicl_lang]} {key.split("_")[0].capitalize()}' for key in src_keys
            ] if ITC_TYPE == 'cross' else []),
            target_keys=tgt_keys, label_keys=['label'], 
            source_keys=src_keys if ITC_TYPE == 'cross' else [], 
            query_keys=query_keys
        )

        # Retrieve & preprocess labels
        label_names = list(set(eval_dset['label']))
        if LABEL_TYPE == 'Target':
            label_map = ioa_df.loc[dset_lang, 'label_map']
        else:
            label_map = None
        
        ###
        # Indexing
        ###
        if ITC_TYPE == 'cross':
            itc_dset = itc_dsets[dset_lang]    
            itc_indexer = DatasetIndexer(dataset=itc_dset, index_key=x_iia_keys, index_type=ITC_INDEX_TYPE)
        elif ITC_TYPE == 'mono':
            itc_dset = itc_dsets[dset_lang]    
            itc_indexer = DatasetIndexer(dataset=itc_dset, index_key=iia_keys, index_type=ITC_INDEX_TYPE)
        else:
            itc_dset = None
            itc_indexer = None
            
        ###
        # Inference
        ###
            
        inputs, preds, golds = [], [], []

        # Check saved data
        if exists(f'outputs_itc/itc-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv'):
            print("Output exist, use partial log instead")
            with open(f'outputs_itc/itc-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    inputs.append(row["Input"])
                    preds.append(row["Pred"])
                    golds.append(row["Gold"])
            print(f"Skipping until {len(preds)}")

        # Perform Inference
        prompts, labels = [], []
        if len(preds) < len(eval_dset):
            for e, sample in enumerate(tqdm(eval_dset)):
                if e < len(preds):
                    continue

                if type(icl_keys) == str:
                    input_query = sample[icl_keys]
                else: # type(icl_keys) == list
                    input_query = [sample[key] for key in icl_keys]
                label = sample['label']
                
                ###
                # Retrieve Exemplars
                ###
                
                # Retrieve ITC Exemplars
                if itc_indexer is not None:
                    itc_samples = itc_indexer.get_similar_samples(input_query, n_samples=ITC_EXEMPLAR_COUNT)
                    if LABEL_TYPE == 'Target':
                        itc_samples['label'] = [label_map[label] for label in itc_samples['label']]
                else:
                    itc_samples = None
                    
                ###
                # Prepare Zero-Shot / Few-Shot Prompt Text
                ###
                prompt_text = itc_prompter.generate_prompt(
                    input_exemplar=sample,
                    exemplars=itc_samples
                )
                
                prompts.append(prompt_text)
                labels.append(label)

                ###
                # Perform zero-shot / few-shot Inference
                ###
                
                # Batch Inference
                if len(prompts) == BATCH_SIZE:
                    if LABEL_TYPE == 'Target':
                        # Map label names from original label to target language label using label_map
                        x_label_names = [label_map[label] for label in label_names]
                        out = predict_classification_batch(model, tokenizer, prompts, x_label_names)
                    else:
                        out = predict_classification_batch(model, tokenizer, prompts, label_names)
                    hyps = torch.argmax(torch.stack(out, dim=-1), dim=-1).tolist()
                    for (prompt, hyp, label) in zip(prompts, hyps, labels):
                        inputs.append(prompt)
                        preds.append(label_names[int(hyp)])
                        golds.append(label)
                    # print(f'label_names[int(hyp)]: ' + ', '.join([label_names[int(hyp)] for hyp in hyps]))
                    # print(f'labels: ' + ', '.join(labels))
                    prompts, labels = [], []                    

                # partial saving
                if len(preds) % (5 * BATCH_SIZE) == 0:
                    inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
                    inference_df.to_csv(f'outputs_itc/itc-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv', index=False)
                   
            # Perform zero-shot / few-shot Inference on last remaining batch data
            if len(prompts) > 0:
                out = predict_classification_batch(model, tokenizer, prompts, label_names)
                hyps = torch.argmax(torch.stack(out, dim=-1), dim=-1).tolist()
                for (prompt, hyp, label) in zip(prompts, hyps, labels):
                    inputs.append(prompt)
                    preds.append(label_names[int(hyp)])
                    golds.append(label)
                # print(f'label_names[int(hyp)]: ' + ', '.join([label_names[int(hyp)] for hyp in hyps]))
                # print(f'labels: ' + ', '.join(labels))
                prompts, labels = [], []
                
        # Full saving
        inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
        inference_df.to_csv(f'outputs_itc/itc-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv', index=False)

        cls_report = classification_report(golds, preds, output_dict=True)
        acc, macro_f1, weighted_f1 = cls_report['accuracy'], cls_report['macro avg']['f1-score'], cls_report['weighted avg']['f1-score']
        print(f'{DATASET_NAME} {dset_lang}')
        print('accuracy', acc)
        print('f1 macro', macro_f1)
        print('f1 weighted', weighted_f1)
        print("===\n\n")

        metrics['dataset'].append(DATASET_NAME)
        metrics['lang'].append(dset_lang)
        metrics['accuracy'].append(acc)
        metrics['macro_f1'].append(macro_f1)
        metrics['weighted_f1'].append(weighted_f1)

    pd.DataFrame.from_dict(metrics).T.reset_index().to_csv(f'metrics_itc/results_{DATASET_NAME}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv', index=False)