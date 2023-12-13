#!/bin/bash

# main_icl_alignment_translate.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <sbert_type> <mt_type> <batch_size>

###
# Zero-Shot => Zero-Shot (Source)
###
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment_translate.py bigscience/bloom-7b1 americasnli none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment_translate.py bigscience/bloom-7b1 americasnli-spa none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment_translate.py bigscience/bloom-7b1 nusatranslation none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment_translate.py bigscience/bloom-7b1 nusatranslation-ind none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment_translate.py bigscience/bloom-7b1 masakhanews none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=0 python main_icl_alignment_translate.py bigscience/bloom-7b1 tweetsentimulti none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8

###
# ICL SBERT
###
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py bigscience/bloom-7b1 americasnli translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py bigscience/bloom-7b1 americasnli-spa translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py bigscience/bloom-7b1 nusatranslation translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py bigscience/bloom-7b1 nusatranslation-ind translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py bigscience/bloom-7b1 masakhanews translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py bigscience/bloom-7b1 tweetsentimulti translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4