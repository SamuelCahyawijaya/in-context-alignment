#!/bin/bash

# main_icl_alignment_sbert.py <model_path_or_name> <dataset_name> <icl_type> <icl_index_type> <icl_num_exemplar> <iia_type> <iia_index_type> <iia_num_exemplar> <ioa_type> <batch_size>

###
# LaBSE
###

# Mono SBERT => Monolingual ICL (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli mono sbert 3 none random 0 False after sentence-transformers/LaBSE 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa mono sbert 3 none random 0 False after sentence-transformers/LaBSE 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation mono sbert 3 none random 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind mono sbert 3 none random 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews mono sbert 3 none random 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti mono sbert 3 none random 0 False after sentence-transformers/LaBSE 4

# Cross SBERT => Cross-Lingual ICL (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli cross sbert 3 none random 0 False after sentence-transformers/LaBSE 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa cross sbert 3 none random 0 False after sentence-transformers/LaBSE 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation cross sbert 3 none random 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind cross sbert 3 none random 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews cross sbert 3 none random 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti cross sbert 3 none random 0 False after sentence-transformers/LaBSE 4

# Cross XPresso => Cross-Lingual Xpresso (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli xpresso sbert 3 none none 0 False after sentence-transformers/LaBSE 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa xpresso sbert 3 none none 0 False after sentence-transformers/LaBSE 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation xpresso sbert 3 none none 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind xpresso sbert 3 none none 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews xpresso sbert 3 none none 0 False after sentence-transformers/LaBSE 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti xpresso sbert 3 none none 0 False after sentence-transformers/LaBSE 4

###
# paraphrase-multilingual-mpnet-base-v2
###

# Mono SBERT => Monolingual ICL (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4

# Cross SBERT => Cross-Lingual ICL (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4

# Cross XPresso => Cross-Lingual Xpresso (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-mpnet-base-v2 4

###
# paraphrase-xlm-r-multilingual-v1
###

# Mono SBERT => Monolingual ICL (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli mono sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa mono sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation mono sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind mono sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews mono sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti mono sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4

# Cross SBERT => Cross-Lingual ICL (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli cross sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa cross sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation cross sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind cross sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews cross sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti cross sbert 3 none random 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4

# Cross XPresso => Cross-Lingual Xpresso (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-xlm-r-multilingual-v1 4

###
# paraphrase-multilingual-MiniLM-L12-v2
###

# Mono SBERT => Monolingual ICL (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti mono sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4

# Cross SBERT => Cross-Lingual ICL (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti cross sbert 3 none random 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4

# Cross XPresso => Cross-Lingual Xpresso (S) (Source)
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B americasnli-spa xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 8
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B nusatranslation-ind xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B masakhanews xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4
CUDA_VISIBLE_DEVICES=1 python main_icl_alignment_sbert.py facebook/xglm-7.5B tweetsentimulti xpresso sbert 3 none none 0 False after sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 4