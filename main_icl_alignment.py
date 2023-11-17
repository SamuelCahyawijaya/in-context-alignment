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
    'nusatranslation-ind': (
        'Prediksikan label sentimen dari kalimat berikut:\n{context}\n{query}',
        '{text} => {label}',
        '{text_1} => {text_2}',
        'text', 'text_1', 'text_2'
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
        raise ValueError('main_icl_alignment.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <iia_type> <iia_index_type> <iia_num_exemplar> <ioa_type> <batch_size>')

    BASE_PATH='./dataset'
    MODEL = sys.argv[1]
    DATASET_NAME = sys.argv[2] # americasnli, nusatranslation, masakhanews
    ICL_TYPE = sys.argv[3] # cross, mono, none
    ICL_INDEX_TYPE = sys.argv[4].split(',') # random, count, tf-idf, sbert
    ICL_EXEMPLAR_COUNT = int(sys.argv[5])
    IIA_TYPE = sys.argv[6] # cross, mono, none
    IIA_INDEX_TYPE = sys.argv[7].split(',') # random, count, tf-idf, sbert
    IIA_EXEMPLAR_COUNT = int(sys.argv[8])
    IOA_TYPE = sys.argv[9]
    BATCH_SIZE= int(sys.argv[10])
    
    SAVE_NAME = f'icl-{ICL_TYPE}-{"$".join(ICL_INDEX_TYPE)}-{ICL_EXEMPLAR_COUNT}_iia-{IIA_TYPE}-{"$".join(IIA_INDEX_TYPE)}-{IIA_EXEMPLAR_COUNT}_ioa-{IOA_TYPE}'

    os.makedirs('./metrics', exist_ok=True) 
    os.makedirs('./outputs', exist_ok=True) 

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
        if IOA_TYPE in ['True', 'Target'] and dset_lang not in ioa_df.index:
            continue

        print(f'Processing {DATASET_NAME} {dset_lang}')

        ###
        # Preprocessing
        ###

        # Extract Metadata
        prompt_template, icl_template, iia_template, icl_keys, iia_keys, x_iia_keys = dataset_to_metadata_map[DATASET_NAME]
        
        # Prepare Prompter
        icl_prompter = ICLPrompter(
            prompt_template=prompt_template, icl_template=icl_template, iia_template=iia_template
        )

        # Retrieve & preprocess labels
        label_names = list(set(eval_dset['label']))
        if IOA_TYPE in ['True', 'Target']:
            label_map = ioa_df.loc[dset_lang, 'label_map']
        else:
            label_map = None
        
        ###
        # Indexing
        ###
        if ICL_TYPE == 'cross':
            icl_dset = icl_dsets[xicl_lang]    
            icl_indexer = DatasetIndexer(dataset=icl_dset, index_key=icl_keys, index_type=ICL_INDEX_TYPE)
        elif ICL_TYPE == 'mono':
            icl_dset = icl_dsets[dset_lang]    
            icl_indexer = DatasetIndexer(dataset=icl_dset, index_key=icl_keys, index_type=ICL_INDEX_TYPE)
        else:
            icl_dset = None
            icl_indexer = None
            
        sbert=None
        if icl_indexer is not None:
            sbert = icl_indexer.sbert
            
        if IIA_TYPE == 'cross':
            iia_dset = iia_dsets[dset_lang]
            iia_indexer = DatasetIndexer(dataset=iia_dset, index_key=x_iia_keys, index_type=IIA_INDEX_TYPE, sbert=sbert)
        elif IIA_TYPE == 'mono':
            iia_dset = iia_dsets[dset_lang]
            iia_indexer = DatasetIndexer(dataset=iia_dset, index_key=iia_keys, index_type=IIA_INDEX_TYPE, sbert=sbert)
        else:
            iia_dset = None
            iia_indexer = None
            
        ###
        # Inference
        ###
            
        inputs, preds, golds = [], [], []

        # Check saved data
        if exists(f'outputs/icl-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv'):
            print("Output exist, use partial log instead")
            with open(f'outputs/icl-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv') as csvfile:
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
                
                # Retrieve ICL Exemplars

                if icl_indexer is not None:
                    icl_samples = icl_indexer.get_similar_samples(input_query, n_samples=ICL_EXEMPLAR_COUNT)
                else:
                    icl_samples = None
                    
                # Retrieve IIA Exemplars
                if iia_indexer is not None:
                    iia_samples = iia_indexer.get_similar_samples(input_query, n_samples=IIA_EXEMPLAR_COUNT)
                else:
                    iia_samples = None

                if IOA_TYPE == 'True':
                    label_prompts = [f"{label} means {label_map[label]}" for label in label_names]
                    label_prompts[-1] = f'and {label_prompts[-1]}'
                    ioa_prompt = f'In {lang_map[dset_lang]} {", ".join(label_prompts) if len(label_prompts) > 2 else " ".join(label_prompts)}'
                elif IOA_TYPE == 'Target':
                    for i in range(ICL_EXEMPLAR_COUNT):
                        icl_samples['label'][i] = label_map[icl_samples['label'][i]]
                    ioa_prompt = None
                else:
                    ioa_prompt = None
                    
                ###
                # Prepare Zero-Shot / Few-Shot Prompt Text
                ###
                prompt_text = icl_prompter.generate_prompt(
                    input_exemplar=sample,
                    icl_exemplars=icl_samples,
                    input_alignment_exemplars=iia_samples,
                    output_alignment_prompt=ioa_prompt
                )
                    
                # print(f'input_query: ' + str(input_query))
                # print(f'label: ' + label)
                # print(f'ioa_prompt: ' + ioa_prompt)
                # print(f'prompt_text:\n' + prompt_text)
                
                prompts.append(prompt_text)
                labels.append(label)

                ###
                # Perform zero-shot / few-shot Inference
                ###
                
                # Batch Inference
                if len(prompts) == BATCH_SIZE:
                    if IOA_TYPE in ['True', 'Target']:
                        # Map label names from original label to target language label using label_map
                        ioa_label_names = [label_map[label] for label in label_names]
                        out = predict_classification_batch(model, tokenizer, prompts, ioa_label_names)
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
                    inference_df.to_csv(f'outputs/icl-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv', index=False)
                   
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
        inference_df.to_csv(f'outputs/icl-alignment_{DATASET_NAME}_{dset_lang}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv', index=False)

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

    pd.DataFrame.from_dict(metrics).T.reset_index().to_csv(f'metrics/results_{DATASET_NAME}_{MODEL.split("/")[-1]}_{SAVE_NAME}.csv', index=False)