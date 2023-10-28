#!/bin/bash

# main_input_aligner.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <iaa_type> <iia_index_type> <iia_num_exemplar> <include_iio> <batch_size>

###
# Zero-Shot
###
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 none random 0 False 16
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 none random 0 False 24
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 none random 0 False 24

###
# ICL Only
###

# Mono Count
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli mono count 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation mono count 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews mono count 3 none random 0 False 12

# Mono TF-IDF
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli mono tf-idf 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation mono tf-idf 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews mono tf-idf 3 none random 0 False 12

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli mono sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation mono sbert 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews mono sbert 3 none random 0 False 12

# Mono Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli mono count,tf-idf,sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation mono count,tf-idf,sbert 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews mono count,tf-idf,sbert 3 none random 0 False 12

# Cross Count
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli cross count 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation cross count 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews cross count 3 none random 0 False 12

# Cross TF-IDF
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli cross tf-idf 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation cross tf-idf 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews cross tf-idf 3 none random 0 False 12

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli cross sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation cross sbert 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews cross sbert 3 none random 0 False 12

# Cross Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli cross count,tf-idf,sbert 3 none random 0 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation cross count,tf-idf,sbert 3 none random 0 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews cross count,tf-idf,sbert 3 none random 0 False 12

###
# IIA Only
###

# Mono Count
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 mono count 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 mono count 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 mono count 3 False 12

# Mono TF-IDF
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 mono tf-idf 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 mono tf-idf 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 mono tf-idf 3 False 12

# Mono SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 mono sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 mono sbert 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 mono sbert 3 False 12

# Mono Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 mono count,tf-idf,sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 mono count,tf-idf,sbert 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 mono count,tf-idf,sbert 3 False 12

# Cross Count
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 cross count 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 cross count 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 cross count 3 False 12

# Cross TF-IDF
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 cross tf-idf 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 cross tf-idf 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 cross tf-idf 3 False 12

# Cross SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 cross sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 cross sbert 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 cross sbert 3 False 12

# Cross Count, TF-IDF, SBERT
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 americasnli none random 0 cross count,tf-idf,sbert 3 False 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 nusatranslation none random 0 cross count,tf-idf,sbert 3 False 12
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment.py bigscience/bloomz-7b1 masakhanews none random 0 cross count,tf-idf,sbert 3 False 12

# IOA Only


# ICL + IIA


# ICL + IOA


# IIA + IOA


# ICL + IIA + IOA

# python main_input_aligner.py bigscience/mt0-small americasnli random mono mono count tf-idf 3 3 False
# python main_input_aligner.py bigscience/mt0-small nusatranslation mono cross random sbert 2 2 True
# python main_input_aligner.py bigscience/mt0-small masakhanews cross cross count,tf-idf,sbert count,tf-idf 1 1 True
