#!/bin/bash

# main_icl_alignment_translate.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <sbert_type> <mt_type> <batch_size>

###
# Zero-Shot => Zero-Shot (Source)
###
# CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_translate.py facebook/xglm-7.5B americasnli none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_translate.py facebook/xglm-7.5B americasnli-spa none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_translate.py facebook/xglm-7.5B nusatranslation none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_translate.py facebook/xglm-7.5B nusatranslation-ind none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_translate.py facebook/xglm-7.5B masakhanews none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8
# CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_translate.py facebook/xglm-7.5B tweetsentimulti none random 0 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 8

###
# ICL SBERT
###
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py facebook/xglm-7.5B americasnli translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py facebook/xglm-7.5B americasnli-spa translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py facebook/xglm-7.5B nusatranslation translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py facebook/xglm-7.5B nusatranslation-ind translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py facebook/xglm-7.5B masakhanews translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4
CUDA_VISIBLE_DEVICES=2 python main_icl_alignment_translate.py facebook/xglm-7.5B tweetsentimulti translate sbert 3 sentence-transformers/stsb-xlm-r-multilingual facebook/nllb-200-distilled-1.3B 4