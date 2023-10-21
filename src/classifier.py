import os, sys
from os.path import exists
import glob
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

@torch.inference_mode()
def get_logprobs(model, tokenizer, inputs, label_ids=None, label_attn=None):
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')
    input_ids, output_ids, attn_mask = inputs["input_ids"][:,:-1], inputs["input_ids"][:, 1:], inputs['attention_mask'][:,:-1]
    
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=output_ids)
    logits = outputs.logits
    
    if model.config.is_encoder_decoder:
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, label_ids.unsqueeze(2)).squeeze(dim=-1) * label_attn
        return (logprobs.squeeze(dim=-1) / label_attn.sum(dim=-1, keepdims=True)).sum(dim=-1).cpu().detach().numpy()
    else:
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)).squeeze(dim=-1)
        logprobs[input_ids == tokenizer.pad_token_id] = 0
        num_tokens = (input_ids != tokenizer.pad_token_id).sum(dim=-1, keepdims=True)
        return (logprobs.squeeze(dim=-1) / num_tokens).sum(dim=1).cpu().detach().numpy()

@torch.inference_mode()
def predict_classification(model, tokenizer, prompts, labels):
    if model.config.is_encoder_decoder:
        labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
        list_label_ids = labels_encoded['input_ids'].to('cuda')
        list_label_attn = labels_encoded['attention_mask'].to('cuda')
        
        inputs = [prompt.replace('[LABELS_CHOICE]', '') for prompt in prompts]
        probs = []
        for (label_ids, label_attn) in zip(list_label_ids, list_label_attn):
            probs.append(
                get_logprobs(model, tokenizer, inputs, label_ids.view(1,-1), label_attn.view(1,-1))
            )
    else:
        probs = []
        for label in labels:
            inputs = []
            for prompt in prompts:
                inputs.append(prompt.replace('[LABELS_CHOICE]', label))
            probs.append(get_logprobs(model, tokenizer, inputs))
    return probs