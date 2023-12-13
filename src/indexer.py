import re
import numpy as np
from datasets import Dataset
from dataset_utils import load_dataset
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import Counter

class SimpleDatasetIndexer(object):
    def __init__(self, 
                 dataset: Dataset, index_key: str | list[str],  index_type: str | list[str], 
                 sbert: SentenceTransformer | None = None, sbert_type: str = 'sentence-transformers/stsb-xlm-r-multilingual'
        ) -> None:
        # dataset: HF datasets.Dataset object to be indexed
        # index_key: The name of the column of the dataset used for the index (only support single column for now)
        # index_type: Type of the index (`random`, `count`, `tf-idf`, `sbert`). 
        #             Scoring-based index (`count`, `tf-idf`, `sbert`) can be aggregated by providing list[str]
        # sbert: SentenceTransformer used for inferencing
        self.dataset = dataset
        if type(index_key) is str:
            self.index_key = [index_key]
        else:
            self.index_key = index_key
            
        if type(index_type) is str:
            self.index_type = [index_type]
        else:
            self.index_type = index_type
            
        if 'sbert' in self.index_type:
            if sbert is not None:
                self.sbert = sbert
            else:
                self.sbert = SentenceTransformer(sbert_type)
        else:
            self.sbert = None
            
        self._index_dataset()

    @torch.no_grad()
    def _index_dataset(self) -> bool:
        # Generate Indices
        if 'random' in self.index_type:
            return
        
        text_corpus = []
        for i in range(len(self.dataset)):
            text = []
            for key in self.index_key:
                 text.append(self.dataset[i][key])
            text_corpus.append('. '.join(text))
        if 'unique' in self.index_type:
            self.word_sets = []
            for text in text_corpus:
                text = re.sub(r'[^\w\s]', '', text)
                tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
                self.word_sets.append(set(tokens))
        if 'count' in self.index_type:
            self.counter = CountVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3, binary=False, lowercase=True)
            self.count_vec = self.counter.fit_transform(text_corpus)
            self.count_sum = np.array(self.count_vec.sum(axis=-1)).squeeze()
        if 'tf-idf' in self.index_type:
            self.tfidfer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3, lowercase=True)
            self.tfidf_vec = self.tfidfer.fit_transform(text_corpus)         
            self.tfidf_sum = np.array(self.tfidf_vec.sum(axis=-1)).squeeze()
        if 'sbert' in self.index_type:
            self.sbert_embeddings = self.sbert.encode(text_corpus, batch_size=128, device='cuda:0', convert_to_tensor=True)
            self.sbert_emb_norm = self.sbert_embeddings.norm(dim=-1)

    def get_similar_samples(self, query: str | dict | list, n_samples: int) -> dict:
        # query: string or list of string or dictionary representing a single instance of sample from the dataset
        # n_samples: number of similar samples to retrieve
        
        # Transform query to stirng
        if type(query) is list:
            query = '. '.join(query)
        elif type(query) is dict:
            text = []
            for key in self.index_key:
                 text.append(query[key])
            query = '. '.join(text)
            
        # Perform indexing
        if 'random' in self.index_type:
            exemplars = []
            indices = np.random.choice(len(self.dataset), size=n_samples, replace=False)
        else:
            if 'unique' in self.index_type:
                query = query.lower()
                # query = re.sub(r'[^\w\s]', '', query)
                tokens = re.findall(r"\w+|[^\w\s]", query, re.UNICODE)
                
                iou_scores = [len(set(tokens).intersection(word_set)) / len(set(tokens).union(word_set)) if len(tokens) > 0 else 0 for word_set in self.word_sets]
                unique_score = torch.softmax(torch.FloatTensor(iou_scores), dim=-1).squeeze().numpy()
            else:
                unique_score = 0
                
            if 'count' in self.index_type:
                sent_count = self.counter.transform([query])
                count_score = np.squeeze(self.count_vec.dot(sent_count.T).toarray()) / (self.count_sum + np.finfo(np.float64).eps)
                count_score = torch.softmax(torch.from_numpy(count_score), dim=-1).squeeze().numpy()
            else:
                count_score = 0
                
            if 'tf-idf' in self.index_type:
                sent_tfidf = self.tfidfer.transform([query])
                tfidf_score = np.squeeze(self.tfidf_vec.dot(sent_tfidf.T).toarray()) / (self.tfidf_sum + np.finfo(np.float64).eps)
                tfidf_score = torch.softmax(torch.from_numpy(tfidf_score), dim=-1).squeeze().numpy()
            else:
                tfidf_score = 0

            if 'sbert' in self.index_type:
                sent_embed = self.sbert.encode([query], device='cuda:0', convert_to_tensor=True)
                sbert_score = torch.softmax(((sent_embed @ self.sbert_embeddings.T) / self.sbert_emb_norm), dim=-1).squeeze().cpu().numpy()
            else:
                sbert_score = 0

            sim_score = (unique_score + count_score + tfidf_score + sbert_score) / 4
            indices = np.argpartition(sim_score, kth=-n_samples, axis=0)[-n_samples:]
                
        # Return topk most similar rows
        return self.dataset[indices]

class XpressoDatasetIndexer(object):
    def __init__(self, 
                 icl_dataset: Dataset, parallel_dataset: Dataset, index_type: str | list[str],
                 parallel_index_key: str | list[str],  parallel_retrieve_key: str | list[str],
                 icl_index_key: str | list[str], sbert_type: str = 'sentence-transformers/stsb-xlm-r-multilingual',
                 sbert: SentenceTransformer | None = None
        ) -> None:
        # icl_dataset: HF datasets.Dataset object to be indexed
        # parallel_dataset: HF datasets.Dataset object to be used as the intermediate parallel data
        # index_key: The name of the column of the dataset used for the index (only support single column for now)
        # parallel_index_key: The name of the column of the parallel dataset used for the index
        # index_type: Type of the index (`random`, `count`, `tf-idf`, `sbert`). 
        #             Scoring-based index (`count`, `tf-idf`, `sbert`) can be aggregated by providing list[str]
        # sbert: SentenceTransformer used for inferencing
        self.icl_dataset = icl_dataset
        self.parallel_dataset = parallel_dataset
        
        if type(index_type) is str:
            self.index_type = [index_type]
        else:
            self.index_type = index_type
            
        if type(parallel_index_key) is str:
            self.parallel_index_key = [parallel_index_key]
        else:
            self.parallel_index_key = parallel_index_key
            
        if type(parallel_retrieve_key) is str:
            self.parallel_retrieve_key = [parallel_retrieve_key]
        else:
            self.parallel_retrieve_key = parallel_retrieve_key

        if type(icl_index_key) is str:
            self.icl_index_key = [icl_index_key]
        else:
            self.icl_index_key = icl_index_key
            
        if sbert is not None:
            self.sbert = sbert
        else:
            self.sbert = SentenceTransformer(sbert_type)
            
        self._index_dataset()

    @torch.no_grad()
    def _index_dataset(self) -> bool:
        # Generate Indices
        if 'random' in self.index_type:
            return
        
        # LRL Data
        text_corpus = []
        for i in range(len(self.parallel_dataset)):
            text = []
            for key in self.parallel_index_key:
                 text.append(self.parallel_dataset[i][key])
            text_corpus.append('. '.join(text))

        if 'unique' in self.index_type:
            self.word_sets = []
            for text in text_corpus:
                text = re.sub(r'[^\w\s]', '', text)
                tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
                self.word_sets.append(set(tokens))
        if 'count' in self.index_type:
            self.counter = CountVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3, binary=False, lowercase=True)
            self.count_vec = self.counter.fit_transform(text_corpus)
            self.count_sum = np.array(self.count_vec.sum(axis=-1)).squeeze()
        if 'tf-idf' in self.index_type:
            self.tfidfer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3, lowercase=True)
            self.tfidf_vec = self.tfidfer.fit_transform(text_corpus)         
            self.tfidf_sum = np.array(self.tfidf_vec.sum(axis=-1)).squeeze()
        if 'sbert' in self.index_type:
            self.lrl_sbert_embeddings = self.sbert.encode(text_corpus, batch_size=128, device='cuda:0', convert_to_tensor=True)
            self.lrl_sbert_emb_norm = self.lrl_sbert_embeddings.norm(dim=-1)
        
        # HRL Data
        text_corpus = []
        for i in range(len(self.icl_dataset)):
            text = []
            for key in self.icl_index_key:
                 text.append(self.icl_dataset[i][key])
            text_corpus.append('. '.join(text))

        self.hrl_sbert_embeddings = self.sbert.encode(text_corpus, batch_size=128, device='cuda:0', convert_to_tensor=True)
        self.hrl_sbert_emb_norm = self.hrl_sbert_embeddings.norm(dim=-1)

    def get_similar_samples(self, query: str | dict | list, n_samples: int) -> dict:
        # query: string or list of string or dictionary representing a single instance of sample from the dataset
        # n_samples: number of similar samples to retrieve
        
        # Transform query to stirng
        if type(query) is list:
            query = '. '.join(query)
        elif type(query) is dict:
            text = []
            for key in self.index_key:
                 text.append(query[key])
            query = '. '.join(text)
            
        # Perform indexing
        if 'random' in self.index_type:
            exemplars = []
            indices = np.random.choice(len(self.icl_dataset), size=n_samples, replace=False)
            
            # Return topk most similar rows
            return self.dataset[indices]
        else:
            # Perform LRL Indexing
            if 'unique' in self.index_type:
                query = query.lower()
                # query = re.sub(r'[^\w\s]', '', query)
                tokens = re.findall(r"\w+|[^\w\s]", query, re.UNICODE)
                
                iou_scores = [len(set(tokens).intersection(word_set)) / len(set(tokens).union(word_set)) if len(tokens) > 0 else 0 for word_set in self.word_sets]
                unique_score = torch.softmax(torch.FloatTensor(iou_scores), dim=-1).squeeze().numpy()
            else:
                unique_score = 0
                
            if 'count' in self.index_type:
                sent_count = self.counter.transform([query])
                count_score = np.squeeze(self.count_vec.dot(sent_count.T).toarray()) / (self.count_sum + np.finfo(np.float64).eps)
                count_score = torch.softmax(torch.from_numpy(count_score), dim=-1).squeeze().numpy()
            else:
                count_score = 0
                
            if 'tf-idf' in self.index_type:
                sent_tfidf = self.tfidfer.transform([query])
                tfidf_score = np.squeeze(self.tfidf_vec.dot(sent_tfidf.T).toarray()) / (self.tfidf_sum + np.finfo(np.float64).eps)
                tfidf_score = torch.softmax(torch.from_numpy(tfidf_score), dim=-1).squeeze().numpy()
            else:
                tfidf_score = 0

            if 'sbert' in self.index_type:
                sent_embed = self.sbert.encode([query], device='cuda:0', convert_to_tensor=True)
                sbert_score = torch.softmax(((sent_embed @ self.lrl_sbert_embeddings.T) / self.lrl_sbert_emb_norm), dim=-1).squeeze().cpu().numpy()
            else:
                sbert_score = 0

            lrl_sim_score = (unique_score + count_score + tfidf_score + sbert_score) / len(self.index_type)
            lrl_indices = np.argpartition(lrl_sim_score, kth=-n_samples, axis=0)[-n_samples:]

            # Perform HRL Indexing
            hrl_indices = []
            hrl_scores = []
            
            for i in range(len(self.parallel_dataset[lrl_indices])):
                text = []
                query = '. '.join([self.parallel_dataset[key][i] for key in self.parallel_retrieve_key])
                text.append(query)

                sent_embed = self.sbert.encode([query], device='cuda:0', convert_to_tensor=True)
                sim_score = torch.softmax(((sent_embed @ self.hrl_sbert_embeddings.T) / self.hrl_sbert_emb_norm), dim=-1).squeeze().cpu().numpy()
                indices = np.argpartition(sim_score, kth=-n_samples, axis=0)[-n_samples:]
                hrl_indices.append(indices)
                hrl_scores.append(sim_score + lrl_sim_score[i])
            hrl_indices, hrl_scores = np.concatenate(hrl_indices), np.concatenate(hrl_scores)
            
            merged_idx_score_map = {}
            for idx, score in zip(hrl_indices, hrl_scores):
                if idx not in merged_idx_score_map:
                    merged_idx_score_map[idx] = 0
                merged_idx_score_map[idx] += score
            hrl_merged_indices = list(map(lambda x: x[0], Counter(merged_idx_score_map).most_common(n_samples)))
            
            # Return topk most similar rows for both ICL and Parallel data
            return self.icl_dataset[hrl_merged_indices], self.parallel_dataset[lrl_indices]

if __name__ == '__main__':
    base_path = '../dataset'
    eval_dsets, _, _, iia_dsets, _, _ = load_dataset('nusatranslation', base_path=base_path)
    dset = eval_dsets['sun']
    parallel_dset = iia_dsets['sun']
    query = 'Hotel ieu atos gadeug ti sajak tilu puluh taun anu kapeungkeur, sateuacan abdi lahir.'
    
    print('QUERY')
    print(query)
    print()
    
    ###
    # Simple Indexer
    ###
    
    # Random Indexer
    print('RANDOM INDEXER')
    indexer = SimpleDatasetIndexer(dataset=dset, index_key='text', index_type='random')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # Unique Indexer
    print('UNIQUE INDEXER')
    indexer = SimpleDatasetIndexer(dataset=dset, index_key='text', index_type='unique')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # Count Indexer
    print('COUNT INDEXER')
    indexer = SimpleDatasetIndexer(dataset=dset, index_key='text', index_type='count')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # TF-IDF Indexer
    print('TF-IDF INDEXER')
    indexer = SimpleDatasetIndexer(dataset=dset, index_key='text', index_type='tf-idf')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # SBERT Indexer
    print('SBERT INDEXER')
    indexer = SimpleDatasetIndexer(dataset=dset, index_key='text', index_type='sbert')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()
    
    # Count + TF-IDF  + SBERT Indexer
    print('UNIQUE + TF-IDF + SBERT INDEXER')
    indexer = SimpleDatasetIndexer(dataset=dset, index_key='text', index_type=['unique', 'tf-idf', 'sbert'])
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # Count + TF-IDF  + SBERT Indexer
    print('UNIQUE + COUNT + TF-IDF + SBERT INDEXER')
    indexer = SimpleDatasetIndexer(dataset=dset, index_key='text', index_type=['unique', 'count', 'tf-idf', 'sbert'])
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    ###
    # XPresso Indexer
    ###

    # XPresso Unique Indexer
    print('XPresso UNIQUE INDEXER')
    indexer = XpressoDatasetIndexer(
        icl_dataset=dset, parallel_dataset=parallel_dset, index_type='unique',
        parallel_index_key='text_1',  parallel_retrieve_key='text_2', icl_index_key='text'
    )
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # XPresso Count Indexer
    print('XPresso COUNT INDEXER')
    indexer = XpressoDatasetIndexer(
        icl_dataset=dset, parallel_dataset=parallel_dset, index_type='count', 
        parallel_index_key='text_1',  parallel_retrieve_key='text_2', icl_index_key='text'
    )
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # XPresso TF-IDF Indexer
    print('XPresso TF-IDF INDEXER')
    indexer = XpressoDatasetIndexer(
        icl_dataset=dset, parallel_dataset=parallel_dset, index_type='tf-idf', 
        parallel_index_key='text_1',  parallel_retrieve_key='text_2', icl_index_key='text'
    )
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # XPresso SBERT Indexer
    print('XPresso SBERT INDEXER')
    indexer = XpressoDatasetIndexer(
        icl_dataset=dset, parallel_dataset=parallel_dset, index_type='sbert', 
        parallel_index_key='text_1',  parallel_retrieve_key='text_2', icl_index_key='text'
    )
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()
    
    # XPresso Count + TF-IDF  + SBERT Indexer
    print('XPresso UNIQUE + TF-IDF + SBERT INDEXER')
    indexer = XpressoDatasetIndexer(
        icl_dataset=dset, parallel_dataset=parallel_dset, index_type=['unique','tf-idf','sbert'],
        parallel_index_key='text_1',  parallel_retrieve_key='text_2', icl_index_key='text'
    )
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # XPresso Unique + Count + TF-IDF  + SBERT Indexer
    print('XPresso UNIQUE + COUNT + TF-IDF + SBERT INDEXER')
    indexer = XpressoDatasetIndexer(
        icl_dataset=dset, parallel_dataset=parallel_dset, index_type=['unique','count','tf-idf','sbert'], 
        parallel_index_key='text_1',  parallel_retrieve_key='text_2', icl_index_key='text'
    )
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()
    
