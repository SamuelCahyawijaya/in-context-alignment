#!/bin/bash

# main_itc_alignment.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <label_type> <batch_size>

###
# Zero-Shot
###
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli none random 0 Source 16
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 Source 16
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews none random 0 Source 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi none random 0 Source 8

###
# Zero-Shot Target Label
###
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli none random 0 Target 16
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 Target 16
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews none random 0 Target 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi none random 0 Target 8

###
# ICL Only
###

# Mono Random
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli mono random 3 Source 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 Source 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 Source 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi mono random 3 Source 4

# Mono SBERT
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 Source 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 Source 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 Source 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi mono sbert 3 Source 4

# Cross SBERT
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 Source 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 Source 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 Source 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi cross sbert 3 Source 4

###
# ICL Target Label
###

# Mono Random Target Label
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli mono random 3 Target 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 Target 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 Target 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi mono random 3 Target 4

# Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 Target 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi mono sbert 3 Target 4

# Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 Target 8
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=3 python main_itc_alignment.py bigscience/bloom-7b1 tweet_senti_multi cross sbert 3 Target 4