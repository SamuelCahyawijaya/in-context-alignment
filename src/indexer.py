from datasets import Dataset
import numpy as np
from dataset_utils import load_dataset
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

class DatasetIndexer(object):
    def __init__(self, dataset: Dataset, index_key: str, index_type: str | list[str]) -> None:
        self.dataset = dataset
        self.index_key = index_key
        if type(index_type) is str:
            self.index_type = [index_type]
        else:
            self.index_type = index_type
            
        self._index_dataset()

    @torch.no_grad()
    def _index_dataset(self) -> bool:
        # Generate Indices
        if 'random' in self.index_type:
            return

        text_corpus = self.dataset[self.index_key]
        if 'count' in self.index_type:
            self.counter = CountVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3, binary=True, lowercase=True)
            self.count_vec = self.counter.fit_transform(text_corpus)
            self.count_sum = np.array(self.count_vec.sum(axis=-1)).squeeze()
        if 'tf-idf' in self.index_type:
            self.tfidfer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3, lowercase=True)
            self.tfidf_vec = self.tfidfer.fit_transform(text_corpus)         
            self.tfidf_sum = np.array(self.tfidf_vec.sum(axis=-1)).squeeze()
        if 'sbert' in self.index_type:
            self.sbert = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
            self.sbert_embeddings = self.sbert.encode(text_corpus, batch_size=128, device='cuda:0', convert_to_tensor=True)
            self.sbert_emb_norm = self.sbert_embeddings.norm(dim=-1)

    def get_similar_samples(self, query: str, n_samples: int) -> dict:
        if 'random' in self.index_type:
            print('RANDOM')
            exemplars = []
            indices = np.random.choice(len(self.dataset), size=n_samples, replace=False)
        else:
            if 'count' in self.index_type:
                print('COUNT')
                sent_count = self.counter.transform([query])
                count_score = np.squeeze(self.count_vec.dot(sent_count.T).toarray()) / (self.count_sum + np.finfo(np.float64).eps)
                count_score = torch.softmax(torch.from_numpy(count_score), dim=-1).squeeze().numpy()
            else:
                count_score = 0
                
            if 'tf-idf' in self.index_type:
                print('TF-IDF')
                sent_tfidf = self.tfidfer.transform([query])
                tfidf_score = np.squeeze(self.tfidf_vec.dot(sent_tfidf.T).toarray()) / (self.tfidf_sum + np.finfo(np.float64).eps)
                tfidf_score = torch.softmax(torch.from_numpy(tfidf_score), dim=-1).squeeze().numpy()
            else:
                tfidf_score = 0

            if 'sbert' in self.index_type:
                print('SBERT')
                sent_embed = self.sbert.encode([query], device='cuda:0', convert_to_tensor=True)
                sbert_score = torch.softmax(((sent_embed @ self.sbert_embeddings.T) / self.sbert_emb_norm), dim=-1).squeeze().numpy()
            else:
                sbert_score = 0

            sim_score = (count_score + tfidf_score + sbert_score) / 3
            indices = np.argpartition(sim_score, kth=-n_samples, axis=0)[-n_samples:]
                
        # Return topk most similar rows
        return self.dataset[indices]

if __name__ == '__main__':
    base_path = '../dataset'
    eval_dsets, context_dsets = load_dataset('nusatranslation', base_path=base_path)
    dset = context_dsets['sun']
    query = 'Hotel ieu atos gadeug ti sajak tilu puluh taun anu kapeungkeur, sateuacan abdi lahir.'
    
    # Random Indexer
    print('RANDOM INDEXER')
    indexer = DatasetIndexer(dataset=dset, index_key='text_1', index_type='random')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # Count Indexer
    print('COUNT INDEXER')
    DatasetIndexer(dataset=dset, index_key='text_1', index_type='count')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # TF-IDF Indexer
    print('TF-IDF INDEXER')
    DatasetIndexer(dataset=dset, index_key='text_1', index_type='tf-idf')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # SBERT Indexer
    print('SBERT INDEXER')
    DatasetIndexer(dataset=dset, index_key='text_1', index_type='sbert')
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # Count + TF-IDF Indexer
    print('COUNT + TF-IDF INDEXER')
    DatasetIndexer(dataset=dset, index_key='text_1', index_type=['count', 'tf-idf'])
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # Count + SBERT Indexer
    print('COUNT + SBERT INDEXER')
    DatasetIndexer(dataset=dset, index_key='text_1', index_type=['count', 'sbert'])
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()

    # TF-IDF + SBERT Indexer
    print('TF-IDF + SBERT INDEXER')
    DatasetIndexer(dataset=dset, index_key='text_1', index_type=['tf-idf', 'sbert'])
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()
    
    # Count + TF-IDF  + SBERT Indexer
    print('COUNT + TF-IDF + SBERT INDEXER')
    DatasetIndexer(dataset=dset, index_key='text_1', index_type=['count', 'tf-idf', 'sbert'])
    print(indexer.get_similar_samples(query=query, n_samples=3))
    print()
