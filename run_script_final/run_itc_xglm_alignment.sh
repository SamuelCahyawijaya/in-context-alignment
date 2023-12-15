#!/bin/bash

# main_itc_alignment.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <label_type> <batch_size>

###
# Zero-Shot
###
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli none random 0 Source 16
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa none random 0 Source 16
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation none random 0 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind none random 0 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews none random 0 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti none random 0 Source 16

###
# Zero-Shot Target Label
###
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli none random 0 Target 16
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa none random 0 Target 16
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews none random 0 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti none random 0 Target 16

###
# ICL Only
###

# Mono Random
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono random 3 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono random 3 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono random 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono random 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono random 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono random 3 Source 8

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono sbert 3 Source 8

# Mono-Trans Random
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono-trans random 3 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono-trans random 3 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono-trans random 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono-trans random 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono-trans random 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono-trans random 3 Source 8

# Mono-Trans SBERT
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono-trans sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono-trans sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono-trans sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono-trans sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono-trans sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono-trans sbert 3 Source 8

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli cross sbert 3 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa cross sbert 3 Source 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation cross sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind cross sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews cross sbert 3 Source 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti cross sbert 3 Source 8

###
# ICL Target Label
###

# Mono Random Target Label
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono random 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono random 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono random 3 Target 8

# Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono sbert 3 Target 8

# Mono-Trans Random Target Label
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono-tran random 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono-tran random 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono-tran random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono-tran random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono-tran random 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono-tran random 3 Target 8

# Mono-Trans SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli mono-trans sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa mono-trans sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation mono-trans sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind mono-trans sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews mono-trans sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti mono-trans sbert 3 Target 8

# Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli cross sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B americasnli-spa cross sbert 3 Target 8
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B nusatranslation-ind cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B masakhanews cross sbert 3 Target 4
CUDA_VISIBLE_DEVICES=1 python main_itc_alignment.py facebook/xglm-7.5B tweetsentimulti cross sbert 3 Target 8