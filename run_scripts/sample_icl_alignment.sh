#!/bin/bash

# icl_sample_preview.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <iia_type> <iia_index_type> <iia_num_exemplar> <ioa_type> <alignment_position> <batch_size>

# ###
# # Zero-Shot
# ###
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 False after 1

# ###
# # Zero-Shot Target Label
# ###
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 Target after 1

# ###
# # ICL Only
# ###

# # Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 False after 1

# # Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 False after 1

# ###
# # IIA Only
# ###

# # Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 False after 1

# # Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 cross sbert 3 False after 1

# ###
# # ICL + IIA
# ###

# # Mono-Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 False after 1

# # Mono-Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 False after 1

# # Cross-Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 False after 1

# # Cross-Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 False after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 False after 1


# ###
# # ICL / IAA / ICL + IAA Target Label
# ###

# ## ICL Target Label

# # Mono SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 Target after 1

# # Cross SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 Target after 1

# ## IAA Target Label

# # Mono Random Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 mono random 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 mono random 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 mono random 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono random 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 mono random 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono random 3 Target after 1

# # Mono SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 Target after 1

# # Cross SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 cross sbert 3 Target after 1

# ## ICL + IAA Target Label

# # Mono-Mono SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 Target after 1

# # Mono-Cross SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 Target after 1

# # Cross-Mono SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 Target after 1

# # Cross-Cross SBERT Target Label
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 Target after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 Target after 1

# ###
# # IOA Only
# ###

# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 none random 0 True after 1

# ###
# # ICL + IOA
# ###

# # Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 True after 1

# # Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 True after 1

# ###
# # IIA + IOA
# ###

# # Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 mono sbert 3 True after 1

# # Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli none random 0 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa none random 0 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation none random 0 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind none random 0 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews none random 0 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti none random 0 cross sbert 3 True after 1

# ###
# # ICL + IIA + IOA
# ###

# # Mono-Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 True after 1

# # Mono-Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 True after 1

# # Cross-Mono SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 True after 1

# # Cross-Cross SBERT
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 True after 1
# CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 True after 1

#####
###
# Alignment Before
###
#####

###
# ICL + IIA
###

# Mono-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 False before 1

# Mono-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 False before 1

# Cross-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 False before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 False before 1

# Cross-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 False after 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 False after 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 False after 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 False after 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 False after 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 False after 1


###
# ICL + IAA Target Label
###

## ICL + IAA Target Label

# Mono-Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 Target before 1

# Mono-Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 Target before 1

# Cross-Mono SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 Target before 1

# Cross-Cross SBERT Target Label
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 Target before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 Target before 1

###
# ICL + IOA
###

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 none random 0 True before 1

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 none random 0 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 none random 0 True before 1

###
# ICL + IIA + IOA
###

# Mono-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 mono sbert 3 True before 1

# Mono-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli mono sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa mono sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation mono sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind mono sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews mono sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti mono sbert 3 cross sbert 3 True before 1

# Cross-Mono SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 mono sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 mono sbert 3 True before 1

# Cross-Cross SBERT
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli cross sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 americasnli-spa cross sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation cross sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 nusatranslation-ind cross sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 masakhanews cross sbert 3 cross sbert 3 True before 1
CUDA_VISIBLE_DEVICES=1 python icl_sample_preview.py bigscience/bloom-7b1 tweetsentimulti cross sbert 3 cross sbert 3 True before 1