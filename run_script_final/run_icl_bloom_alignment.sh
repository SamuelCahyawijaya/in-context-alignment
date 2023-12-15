#!/bin/bash

# main_icl_alignment.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <iia_type> <iia_index_type> <iia_num_exemplar> <ioa_type> <batch_size>

# ###
# # Zero-Shot => Zero-Shot (Source)
# ###
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 none random 0 False after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 False after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 False after 8

# ###
# # Zero-Shot Target Label => Zero-Shot (Target)
# ###
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 none random 0 Target after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 Target after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 Target after 8

###
# ICL Only
###

# # Mono Random => Monolingual ICL (R) (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 none random 0 False after 4

# # Mono Unique => Monolingual ICL (U) (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono unique 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono unique 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono unique 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono unique 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono unique 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono unique 3 none random 0 False after 4

# # Mono TF-IDF => Monolingual ICL (T) (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono tf-idf 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono tf-idf 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono tf-idf 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono tf-idf 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono tf-idf 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono tf-idf 3 none random 0 False after 4

# # Mono SBERT => Monolingual ICL (S) (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 False after 4

# # Mono Unique, TF-IDF, SBERT => Monolingual ICL (UTS) (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono unique,tf-idf,sbert 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono unique,tf-idf,sbert 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono unique,tf-idf,sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono unique,tf-idf,sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono unique,tf-idf,sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono unique,tf-idf,sbert 3 none random 0 False after 4

# # Cross SBERT => Cross-Lingual ICL (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 False after 4

# # Cross XPresso => Cross-Lingual Xpresso (S) (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli xpresso sbert 3 none none 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa xpresso sbert 3 none none 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation xpresso sbert 3 none none 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind xpresso sbert 3 none none 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews xpresso sbert 3 none none 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti xpresso sbert 3 none none 0 False after 4

# # Cross XPresso => Cross-Lingual Xpresso (UTSS) (Source)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli xpresso unique,tf-idf,sbert 3 none none 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa xpresso unique,tf-idf,sbert 3 none none 0 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation xpresso unique,tf-idf,sbert 3 none none 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind xpresso unique,tf-idf,sbert 3 none none 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews xpresso unique,tf-idf,sbert 3 none none 0 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti xpresso unique,tf-idf,sbert 3 none none 0 False after 4

###
# IIA Only
###

# # Mono SBERT => Input Alignment (New Prompt) (S)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 False after 4

# # Mono Unique, TF-IDF, SBERT => Input Alignment (New Prompt) (UTS)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono unique,tf-idf,sbert 3 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono unique,tf-idf,sbert 3 False after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono unique,tf-idf,sbert 3 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono unique,tf-idf,sbert 3 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono unique,tf-idf,sbert 3 False after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono unique,tf-idf,sbert 3 False after 4

###
# ICL + IIA
###

# Cross + Mono Unique, TF-IDF, SBERT => ICL + Input Alignment (New Prompt) (UTS)
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 mono unique,tf-idf,sbert 3 False after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono unique,tf-idf,sbert 3 False after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono unique,tf-idf,sbert 3 False after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono unique,tf-idf,sbert 3 False after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono unique,tf-idf,sbert 3 False after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono unique,tf-idf,sbert 3 False after 4

###
# ICL + IIA + IOA
###

# Cross + Mono Unique, TF-IDF, SBERT + IOA => ICL + Input Alignment (New Prompt) (UTS) + Output Alignment
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 mono unique,tf-idf,sbert 3 True after 8
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono unique,tf-idf,sbert 3 True after 8
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono unique,tf-idf,sbert 3 True after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono unique,tf-idf,sbert 3 True after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono unique,tf-idf,sbert 3 True after 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono unique,tf-idf,sbert 3 True after 4

###
# ICL / IIA / ICL + IIA Target Label
###

# # ICL Target Label

# # Mono Random Target Label => Monolingual ICL (R) (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono random 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono random 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono random 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono random 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono random 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono random 3 none random 0 Target after 4

# # Mono Unique Target Label => Monolingual ICL (U) (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono unique 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono unique 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono unique 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono unique 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono unique 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono unique 3 none random 0 Target after 4

# # Mono TF-IDF Target Label => Monolingual ICL (T) (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono tf-idf 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono tf-idf 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono tf-idf 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono tf-idf 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono tf-idf 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono tf-idf 3 none random 0 Target after 4

# # Mono SBERT Target Label => Monolingual ICL (S) (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 Target after 4

# # Mono SBERT Target Label => Monolingual ICL (S) (Target) => Monolingual ICL (UTS) (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli mono unique,tf-idf,sbert 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa mono unique,tf-idf,sbert 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation mono unique,tf-idf,sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind mono unique,tf-idf,sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews mono unique,tf-idf,sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti mono unique,tf-idf,sbert 3 none random 0 Target after 4

# # Cross SBERT Target Label => Cross-Lingual ICL (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 Target after 4

# # Cross XPresso => Cross-Lingual Xpresso (S) (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli xpresso sbert 3 none none 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa xpresso sbert 3 none none 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation xpresso sbert 3 none none 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind xpresso sbert 3 none none 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews xpresso sbert 3 none none 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti xpresso sbert 3 none none 0 Target after 4

# # Cross XPresso => Cross-Lingual Xpresso (UTSS) (Target)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli xpresso unique,tf-idf,sbert 3 none none 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa xpresso unique,tf-idf,sbert 3 none none 0 Target after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation xpresso unique,tf-idf,sbert 3 none none 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind xpresso unique,tf-idf,sbert 3 none none 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews xpresso unique,tf-idf,sbert 3 none none 0 Target after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti xpresso unique,tf-idf,sbert 3 none none 0 Target after 4

# ###
# # IOA Only => Label Alignment
# ###

# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 none random 0 True after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 True after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 True after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 True after 16
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 True after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 True after 8

###
# ICL + IOA
###

# # Cross SBERT => X-Insta (S) (After)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 True after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 True after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 True after 4

###
# IIA + IOA
###

# # Mono SBERT => Input-Output Alignment (New Prompt) (S)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 True after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 True after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 True after 4

# # Mono Unique, TF-IDF, SBERT => Input-Output Alignment (New Prompt) (UTS)
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli none random 0 mono unique,tf-idf,sbert 3 True after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa none random 0 mono unique,tf-idf,sbert 3 True after 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation none random 0 mono unique,tf-idf,sbert 3 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono unique,tf-idf,sbert 3 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none random 0 mono unique,tf-idf,sbert 3 True after 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono unique,tf-idf,sbert 3 True after 4

#####
###
# Alignment Before
###
#####

###
# # ICL + IIA => XPresso (Before)
# ###

# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli xpresso sbert 3 none none 0 False before 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa xpresso sbert 3 none none 0 False before 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation xpresso sbert 3 none none 0 False before 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind xpresso sbert 3 none none 0 False before 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews xpresso sbert 3 none none 0 False before 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti xpresso sbert 3 none none 0 False before 4

# ###
# # ICL + IIA => XPresso (UTS) (Before)
# ###

# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli xpresso unique,tf-idf,sbert 3 none none 0 False before 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa xpresso unique,tf-idf,sbert 3 none none 0 False before 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation xpresso unique,tf-idf,sbert 3 none none 0 False before 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind xpresso unique,tf-idf,sbert 3 none none 0 False before 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews xpresso unique,tf-idf,sbert 3 none none 0 False before 4
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti xpresso unique,tf-idf,sbert 3 none none 0 False before 4

###
# ICL + IOA
###

# # Cross SBERT => X-Insta (Before)
# CUDA_VISIBLE_DEVICES=2 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 True before 8
# CUDA_VISIBLE_DEVICES=2 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 True before 8
# CUDA_VISIBLE_DEVICES=2 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 True before 4
# CUDA_VISIBLE_DEVICES=2 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 True before 4
# CUDA_VISIBLE_DEVICES=2 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 True before 4
# CUDA_VISIBLE_DEVICES=2 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 True before 4

###
# ICL + IIA => Non-Xpresso
###

# Cross + Mono Unique, TF-IDF, SBERT => ICL + Input Alignment (New Prompt) (UTS)
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 mono unique,tf-idf,sbert 3 False before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono unique,tf-idf,sbert 3 False before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono unique,tf-idf,sbert 3 False before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono unique,tf-idf,sbert 3 False before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono unique,tf-idf,sbert 3 False before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono unique,tf-idf,sbert 3 False before 4

###
# ICL + IIA + IOA => Non-Xpresso
###

# Cross + Mono Unique, TF-IDF, SBERT + IOA => ICL + Input Alignment (New Prompt) (UTS) + Output Alignment
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli cross sbert 3 mono unique,tf-idf,sbert 3 True before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono unique,tf-idf,sbert 3 True before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono unique,tf-idf,sbert 3 True before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono unique,tf-idf,sbert 3 True before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 masakhanews none cross sbert 3 unique,tf-idf,sbert 3 True before 4
CUDA_VISIBLE_DEVICES=0 python main_icl_alignment.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono unique,tf-idf,sbert 3 True before 4