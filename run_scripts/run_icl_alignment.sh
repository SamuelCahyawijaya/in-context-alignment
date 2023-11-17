#!/bin/bash

# main_icl_alignment.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <iia_type> <iia_index_type> <iia_num_exemplar> <ioa_type> <batch_size>

###
# Zero-Shot
###
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 none random 0 False 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 False 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 False 8

###
# Zero-Shot Target Label
###
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 none random 0 Target 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 Target 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 Target 8

###
# ICL Only
###

# Mono Random
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 none random 0 False 4

# Mono Count
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono count 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono count 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono count 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono count 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono count 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono count 3 none random 0 False 4

# Mono TF-IDF
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono tf-idf 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono tf-idf 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono tf-idf 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono tf-idf 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono tf-idf 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono tf-idf 3 none random 0 False 4

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 False 4

# Mono Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono count,tf-idf,sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono count,tf-idf,sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono count,tf-idf,sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono count,tf-idf,sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono count,tf-idf,sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono count,tf-idf,sbert 3 none random 0 False 4

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 False 4

# Cross Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross count,tf-idf,sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross count,tf-idf,sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross count,tf-idf,sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross count,tf-idf,sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross count,tf-idf,sbert 3 none random 0 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross count,tf-idf,sbert 3 none random 0 False 4

###
# IIA Only
###

# Mono Random
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono random 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono random 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono random 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono random 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono random 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono random 3 False 4

# Mono Count
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono count 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono count 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono count 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono count 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono count 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono count 3 False 4

# Mono TF-IDF
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono tf-idf 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono tf-idf 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono tf-idf 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono tf-idf 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono tf-idf 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono tf-idf 3 False 4

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 False 4

# Mono Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono count,tf-idf,sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono count,tf-idf,sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono count,tf-idf,sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono count,tf-idf,sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono count,tf-idf,sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono count,tf-idf,sbert 3 False 4

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 cross sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 cross sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 cross sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 cross sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 cross sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 cross sbert 3 False 4

# # Cross Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 cross count,tf-idf,sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 cross count,tf-idf,sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 cross count,tf-idf,sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 cross count,tf-idf,sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 cross count,tf-idf,sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 cross count,tf-idf,sbert 3 False 4

###
# ICL + IIA
###

# Mono-Mono Random
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 mono random 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 mono random 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 mono random 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 mono random 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 mono random 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 mono random 3 False 2

# Mono-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 False 2

# Mono-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 False 2

# Cross-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 False 2

# Cross-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 False 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 False 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 False 2


###
# ICL / IAA / ICL + IAA Target Label
###

## ICL Target Label

# Mono Random Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 none random 0 Target 4

# Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 Target 4

# Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 Target 4

## IAA Target Label

# Mono Random Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono random 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono random 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono random 3 Target 4

# Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 Target 4

# Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 cross sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 cross sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 cross sbert 3 Target 4

## ICL + IAA Target Label

# Mono-Mono Random Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 mono random 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 mono random 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 mono random 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 mono random 3 Target 2

# Mono-Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 Target 2

# Mono-Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 Target 2

# Cross-Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 Target 2

# Cross-Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 Target 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 Target 2

###
# IOA Only
###

CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 none random 0 True 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 True 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 True 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 True 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 True 8

###
# ICL + IOA
###

# Mono Random
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 none random 0 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 none random 0 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 none random 0 True 4

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 True 4

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 True 4

###
# IIA + IOA
###

# Mono Random
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono random 3 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono random 3 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono random 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono random 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono random 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono random 3 True 4

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 True 4

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 cross sbert 3 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 cross sbert 3 True 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 cross sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 cross sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 cross sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 cross sbert 3 True 4

###
# ICL + IIA + IOA
###

# Mono-Mono Random
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 mono random 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 mono random 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 mono random 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 mono random 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 mono random 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 mono random 3 True 2

# Mono-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 True 2

# Mono-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 True 2

# Cross-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 True 2

# Cross-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 True 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 True 2
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 True 2