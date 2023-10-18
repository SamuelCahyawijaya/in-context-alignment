from datasets import Dataset
import numpy as np

class DatasetIndexer(object):
    __slots__ = ("dataset", "index_key", "output_keys", "index_type")

    def __init__(self, dataset: Dataset, index_key: str, output_keys: list[str], index_type: str | list[str]) -> None:
        self.dataset = dataset
        self.index_key = index_key
        self.output_keys = output_keys
        self.index_type = index_type
        
        self._index_dataset()

    @torch.no_grad()
    def _index_dataset(self) -> bool:
        if type(self.index_type) is str:
            self.index_type = [self.index_type]

        # Generate Indices
        text_corpus = self.dataset[index_key]
        if 'count' in self.index_type:
            self.counter = CountVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3, binary=True)
            self.count_vec = counter.fit_transform(text_corpus)
            self.count_sum = np.array(count_vec.sum(axis=-1)).squeeze()
        if 'tf-idf' in self.index_type:
            self.tfidfer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=3)
            self.tfidf_vec = tfidfer.fit_transform(text_corpus)         
            self.tfidf_sum = np.array(tfidf_vec.sum(axis=-1)).squeeze()
        if 'sbert' in self.index_type:
            self.sbert = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
            self.sbert_embeddings = sbert.encode(text_corpus, batch_size=128, device='cuda:0', convert_to_tensor=True)
            self.sbert_emb_norm = self.sbert_embeddings.norm(dim=-1)

    def get_similar_samples(self, query: str, n_samples: int) -> dict:
        if 'random' in self.index_type:
            exemplars = []
            indices = np.random.choice(len(self.dataset), size=n_samples, replace=False)
        else:
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
                sbert_score = torch.softmax(((sent_embed @ self.sbert_embeddings.T) / self.sbert_emb_norm), dim=-1).squeeze().numpy()
            else:
                sbert_score = 0

            sim_score = (count_score + tfidf_score + sbert_score) / 3
            indices = np.argpartition(sim_score, kth=-n_samples, axis=0)[-n_samples:]
                
        # Return topk most similar rows
        return self.dataset[indices]
