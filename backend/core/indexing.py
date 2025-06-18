from ..core.structures import PostingsLinkedList
from ..core.utils import read_data, process_row, preprocess
from collections import defaultdict
import math
from tqdm.auto import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from collections import Counter
import numpy as np


class InvertedIndex:
    def __init__(self, filename):
        self.term_dictionary = defaultdict(
            lambda: {"size_of_postings": 0, "postings_id": None}
        )
        self.postings_store = defaultdict(PostingsLinkedList)
        self.posting_counter = 0
        self.bigram_dictionary = defaultdict(set)
        self.permuterm_dictionary = defaultdict(set)
        self.df = read_data(filename)
        self.idf_vector = None
        self.term_to_index = None
        self.tfidf_matrix = None

    def index(self):
        term_dict = self.term_dictionary
        for terms, docID in process_row(self.df):
            for term in set(terms):
                entry = term_dict[term]
                if entry["postings_id"] is None:
                    # assign an ID to each postings list
                    postings_id = self.posting_counter
                    entry["postings_id"] = postings_id
                    self.posting_counter += 1

                entry["size_of_postings"] += 1
                self.postings_store[entry["postings_id"]].add_posting(docID)

    def bigram_tokenize_term(self, term):
        # convert term "bird" into bigram "$b", "bi", "ir", "rd", "d$"
        if not term or term.isspace():
            return []
        term = "$" + term + "$"
        bigrams = [term[i : i + 2] for i in range(len(term) - 1)]
        return bigrams

    def bigram_index(self):
        # build bigram index of each term
        for term in self.term_dictionary:
            if not term or term.isspace():
                continue
            bigrams = self.bigram_tokenize_term(term)
            for bigram in bigrams:
                self.bigram_dictionary[bigram].add(term)

    def permuterm_index(self):
        # build permuterm index of each term
        for term in self.term_dictionary:
            if not term or term.isspace():
                continue
            rotated_term = term + "$"
            for _ in range(len(rotated_term)):
                self.permuterm_dictionary[rotated_term].add(term)
                rotated_term = rotated_term[1:] + rotated_term[0]

    def build_tfidf_matrix(self):
        vocab = list(self.term_dictionary.keys())
        self.term_to_index = {term: idx for idx, term in enumerate(vocab)}
        num_docs = len(self.df)
        num_terms = len(vocab)

        self.tfidf_matrix = lil_matrix(
            (num_docs, num_terms)
        )  # initialize the tf-idf matrix by lazy insert (shape:[num_docs, num_terms])

        # compute idf vector
        self.idf_vector = np.array(
            [
                math.log(
                    (1 + num_docs)
                    / (1 + self.term_dictionary[term]["size_of_postings"]),
                    10,
                )  # one-plus smoothimg
                for term in vocab
            ]
        )

        # compute the tf of each doc
        for i, row in tqdm(enumerate(self.df.itertuples()), total=len(self.df)):
            tokens = preprocess(row.text)

            filtered_tokens = [t for t in tokens if t in self.term_to_index]
            tf_counts = Counter(
                filtered_tokens
            )  # count the frequency of each term occurring in the doc

            for token, freq in tf_counts.items():
                j = self.term_to_index[token]
                tf = 1 + math.log(freq, 10) if freq > 0 else 0
                self.tfidf_matrix[i, j] = tf * self.idf_vector[j]

        self.tfidf_matrix = self.tfidf_matrix.tocsr()
