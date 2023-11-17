#!/bin/bash

# itc_sample_preview.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <label_type> <batch_size>

###
# Zero-Shot
###
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 Source 1

###
# Zero-Shot Target Label
###
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 Target 1

###
# ICL Only
###

# Mono Random
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono random 3 Source 1

# Mono SBERT
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 Source 1

# Mono-Trans Random
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono-trans random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono-trans random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono-trans random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono-trans random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono-trans random 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono-trans random 3 Source 1

# Mono-Trans SBERT
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono-trans sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono-trans sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono-trans sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono-trans sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono-trans sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono-trans sbert 3 Source 1

# Cross SBERT
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 Source 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 Source 1

###
# ICL Target Label
###

# Mono Random Target Label
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono random 3 Target 1

# Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 Target 1

# Mono-Trans Random
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono-trans random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono-trans random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono-trans random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono-trans random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono-trans random 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono-trans random 3 Target 1

# Mono-Trans SBERT
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli mono-trans sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono-trans sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation mono-trans sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono-trans sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews mono-trans sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono-trans sbert 3 Target 1

# Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 Target 1
CUDA_VISIBLE_DEVICES=2 python itc_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 Target 1