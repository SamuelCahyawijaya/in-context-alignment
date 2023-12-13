import os, sys
from os.path import exists
import glob
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

###
# Single Instance
###
@torch.no_grad()
def get_logprobs(model, tokenizer, prompt, label_ids=None, label_attn=None, device='cuda'):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    
    if model.config.is_encoder_decoder:
        logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, label_ids.unsqueeze(2)) * label_attn.unsqueeze(2)
        return (logprobs.sum() / label_attn.sum()).cpu()
    else:
        logprobs = torch.gather(F.log_softmax(logits, dim=2) * inputs['attention_mask'].unsqueeze(2), 2, output_ids.unsqueeze(2))
        return logprobs.sum().cpu()

def predict_classification(model, tokenizer, prompt, labels, device='cuda'):
    if model.config.is_encoder_decoder:
        labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
        list_label_ids =labels_encoded['input_ids'].to(device)
        list_label_attn =labels_encoded['attention_mask'].to(device)
        probs = [
            get_logprobs(model, tokenizer, prompt.replace('[LABELS_CHOICE]', ''), label_ids.view(1,-1), label_attn.view(1,-1), device=device) 
             for (label_ids, label_attn) in zip(list_label_ids, list_label_attn)
        ]
    else:
        probs = [get_logprobs(model, tokenizer, prompt.replace('[LABELS_CHOICE]', label), device=device) for label in labels]
    return probs

###
# Batching Instance
###
@torch.inference_mode()
def get_logprobs_batch(model, tokenizer, inputs, label_ids=None, label_attn=None, device='cuda'):
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    input_ids, output_ids, attn_mask = inputs["input_ids"][:,:-1], inputs["input_ids"][:, 1:], inputs['attention_mask'][:,:-1]
    
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits
    
    if model.config.is_encoder_decoder:
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, label_ids.unsqueeze(2)).squeeze(dim=-1) * label_attn
        return (logprobs.squeeze(dim=-1)).sum(dim=-1).cpu()
        # return (logprobs.squeeze(dim=-1) / label_attn.sum(dim=-1, keepdims=True)).sum(dim=-1).cpu()
    else:
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)).squeeze(dim=-1)
        logprobs[input_ids == tokenizer.pad_token_id] = 0
        return (logprobs.squeeze(dim=-1)).sum(dim=1).cpu()
        # num_tokens = (input_ids != tokenizer.pad_token_id).sum(dim=-1, keepdims=True)
        # return (logprobs.squeeze(dim=-1) / num_tokens).sum(dim=1).cpu()

@torch.inference_mode()
def predict_classification_batch(model, tokenizer, prompts, labels, device='cuda'):
    if model.config.is_encoder_decoder:
        labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
        list_label_ids = labels_encoded['input_ids'].to(device)
        list_label_attn = labels_encoded['attention_mask'].to(device)
        
        inputs = [prompt.replace('[LABELS_CHOICE]', '') for prompt in prompts]
        probs = []
        for (label_ids, label_attn) in zip(list_label_ids, list_label_attn):
            probs.append(
                get_logprobs_batch(model, tokenizer, inputs, label_ids.view(1,-1), label_attn.view(1,-1), device=device)
            )
    else:
        probs = []
        for label in labels:
            inputs = []
            for prompt in prompts:
                inputs.append(prompt.replace('[LABELS_CHOICE]', label))
            probs.append(get_logprobs_batch(model, tokenizer, inputs, device=device))
    return probs