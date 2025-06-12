from ..core.utils import preprocess
import time
import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class RetrievalEngine:
    def __init__(
        self,
        term_dictionary,
        postings_store,
        df,
        bigram_dictionary,
        permuterm_dictionary,
        term_to_index,
        idf_vector,
        tfidf_matrix,
    ):
        self.term_dictionary = term_dictionary
        self.postings_store = postings_store
        self.df = df
        self.bigram_dictionary = bigram_dictionary
        self.permuterm_dictionary = permuterm_dictionary
        self.term_to_index = term_to_index
        self.idf_vector = idf_vector
        self.tfidf_matrix = tfidf_matrix

    ### ------- Basic Retrieval -------
    def extract_docID(self, postings_list) -> list[int]:
        doc_ids = []
        current = postings_list.head
        while current:
            doc_ids.append(current.docID)
            current = current.next
        return doc_ids

    def intersect(
        self, posting_list1: list[int], posting_list2: list[int]
    ) -> list[int]:
        intersection = []
        iter1 = iter(posting_list1)
        iter2 = iter(posting_list2)
        current1 = next(iter1, None)
        current2 = next(iter2, None)
        while current1 is not None and current2 is not None:
            if current1 == current2:
                intersection.append(current1)
                current1 = next(iter1, None)
                current2 = next(iter2, None)
            elif current1 < current2:
                current1 = next(iter1, None)
            else:
                current2 = next(iter2, None)
        return intersection

    def single_term_query(self, input: str) -> list[int]:
        input = input.lower()
        if input in self.term_dictionary:
            postings_id = self.term_dictionary[input]["postings_id"]
            postings_list = self.postings_store[postings_id]
            return self.extract_docID(postings_list)
        else:
            return []

    def boolean_query(self, input1: str, input2=None):
        postings_list1 = sorted(self.single_term_query(input1))
        if input2 is None:
            return postings_list1
        else:
            postings_list2 = sorted(self.single_term_query(input2))
            return self.intersect(postings_list1, postings_list2)

    def multi_term_intersect(self, doc_lists: list[list[int]]) -> list[int]:
        if not doc_lists:
            return []
        result = doc_lists[0]
        for doc_list in doc_lists[1:]:
            result = self.intersect(result, doc_list)
        return result

    def terms_query(self, terms: list) -> list[int]:
        doc_lists = [sorted(self.single_term_query(t)) for t in terms]
        results = self.multi_term_intersect(doc_lists)
        return results

    ### ------- Wildcard Retrieval (Bigram & Permuterm) -------
    def bigram_parse_wildcard(self, wildcard_query: str) -> list[str]:
        """
        Parse a wildcard query into one or more fragments,
        each of which will later be converted into bigrams.

        Args:
            wildcard_query (str): A wildcard pattern, e.g., "zeit*", "*zeit", "z*it", "*zeit*"

        Returns:
            list[str]: One or more fragments decorated with boundary symbols "$"
        """
        input = wildcard_query.lower()
        query_fragments = []
        if "*" in input:
            chars_list = input.split("*")
            # pattern e.g. "*eit*"
            if chars_list[0] == "" and chars_list[-1] == "":
                query_fragments = [input[1:-1]]
            # pattern e.g. "*eit"
            elif chars_list[0] == "":
                query_fragments = [input[1:] + "$"]
            # pattern e.g. "eit*"
            elif chars_list[-1] == "":
                query_fragments = ["$" + input[:-1]]
            # pattern e.g. "ei*t"
            else:
                query_fragments = ["$" + chars_list[0], chars_list[1] + "$"]
        # pattern e.g. "eit"
        else:
            query_fragments = ["$" + input + "$"]
        return query_fragments

    def bigram_tokenize_query(self, wildcard_query: str) -> list[str]:
        """
        Convert a wildcard query into bigrams for use in bigram indexing.

        Args:
            wildcard_query (str): A query string containing a wildcard (*), e.g., "he*lo"

        Returns:
            list[str]: A list of bigram tokens (e.g., ["$h", "he", "el", "lo", "o$"])
        """
        query_fragments = self.bigram_parse_wildcard(wildcard_query)
        query_bigrams = []
        for elem in query_fragments:
            query_bigrams += [elem[i : i + 2] for i in range(len(elem) - 1)]
        return query_bigrams

    def bigram_get_candidate_terms(self, bigrams: list[str]) -> set[str]:
        """
        Given a list of bigrams, return the set of candidate terms
        that contain all the specified bigrams.

        This performs intersection over the postings (term sets)
        for each bigram from the bigram dictionary.

        Args:
            bigrams (list[str]): List of bigrams from the wildcard query

        Returns:
            set[str]: Candidate terms that contain all input bigrams
        """
        result = set()
        for bigram in bigrams:
            terms = self.bigram_dictionary.get(bigram, set())
            if not result:
                result = terms
            else:
                result = result & terms
        return result

    def permuterm_rotate_wildcard(self, input):
        input = input.lower()
        rotated_input = str()
        if "*" in input:
            chars_list = input.split("*")
            # pattern e.g. "*eit*"
            if chars_list[0] == "" and chars_list[-1] == "":
                rotated_input = input[1:-1]
            # pattern e.g. "*eit"
            elif chars_list[0] == "":
                rotated_input = input[1:] + "$"
            # pattern e.g. "eit*"
            elif chars_list[-1] == "":
                rotated_input = "$" + input[:-1]
            # pattern e.g. "ei*t"
            else:
                rotated_input = chars_list[1] + "$" + chars_list[0]
        else:
            rotated_input = input + "$"
        return rotated_input

    def permuterm_get_candidate_terms(self, rotated_input) -> set[str]:
        result = set()
        if not "$" in rotated_input:
            for key in self.permuterm_dictionary:
                if rotated_input in key:
                    result.update(self.permuterm_dictionary.get(key, set()))
        else:
            for key in self.permuterm_dictionary:
                if key.startswith(rotated_input):
                    result.update(self.permuterm_dictionary.get(key, set()))
        return result

    def get_wildcard_terms(self, input, mode="bigram") -> set[str]:
        if mode == "bigram":
            start = time.time()
            query_bigrams = self.bigram_tokenize_query(input)
            terms = self.bigram_get_candidate_terms(query_bigrams)
            end = time.time()
        elif mode == "permuterm":
            start = time.time()
            rotated_input = self.permuterm_rotate_wildcard(input)
            terms = self.permuterm_get_candidate_terms(rotated_input)
            end = time.time()
        else:
            print("Empty mode argument required.")
            return
        # print(f"{mode} query took {end - start:.4f} seconds")
        return terms

    def wildcard_and_query(self, input, mode=None) -> list[int]:
        input = input.split()
        doc_lists = []

        for i in input:
            if "*" in i:  # process wildcard input
                wildcard_docs = set()
                terms = self.get_wildcard_terms(i, mode=mode)
                for t in terms:
                    wildcard_docs.update(self.single_term_query(t))
                doc_lists.append(sorted(list(wildcard_docs)))
            else:
                doc_lists.append(sorted(self.single_term_query(i)))
        results = self.multi_term_intersect(doc_lists)
        return results

    def show_query_result(self, doc_ids):
        if not doc_ids:
            print("No related documents retrieved.")
            return

        if self.df is None:
            print("No dataframe loaded. Please run index() first.")
            return

        for doc_id in doc_ids:
            row = self.df.loc[self.df["docID"] == doc_id]
            if not row.empty:
                text = row.iloc[0]["text"]
                tweet_id = row.iloc[0]["tweetID"]
                print(f"TweetID: {tweet_id}\nText: {text}\n")

        save = input("Save result as CSV? (y/n): ").strip().lower()
        if save == "y":
            rows = []
            for id in doc_ids:
                row = self.df[self.df["docID"] == id]
                tweet_id = row.iloc[0]["tweetID"]
                text = row.iloc[0]["text"]
                rows.append({"tweet_id": tweet_id, "text": text})
            pd.DataFrame(rows).to_csv("output/query_boolean_result.csv", index=False)
            print("Results saved to query_boolean_result.csv")

    ### ------- TF-IDF Similarity Search -------
    def query_similarity_top_k(self, input, k=100):
        tokens = preprocess(input)
        # compute the query tf-idf vector
        tf_query = np.zeros(len(self.term_to_index))
        for token in tokens:
            if token in self.term_to_index:
                idx = self.term_to_index[token]
                tf_query[idx] += 1

        for idx in range(len(tf_query)):
            if tf_query[idx] != 0:
                tf_query[idx] = 1 + math.log(tf_query[idx], 10)

        tfidf_query = tf_query * self.idf_vector

        # compute cosine similarity
        # norm_query = np.linalg.norm(tfidf_query)
        # norms_docs = np.linalg.norm(self.tfidf_matrix, axis=1)
        # dot_products = self.tfidf_matrix @ tfidf_query
        # similarities = dot_products / (norm_query * norms_docs + 1e-10) # Add small number to avoid invalid denominator

        tfidf_query_sparse = csr_matrix(tfidf_query)

        similarities = cosine_similarity(
            tfidf_query_sparse, self.tfidf_matrix
        ).flatten()

        top_k = similarities.argsort()[-k:][::-1]
        top_doc_ids = self.df.iloc[top_k]["docID"].tolist()
        return list(zip(top_doc_ids, similarities[top_k]))

    def show_top_k_similar(self, results):
        if not results:
            print("No similar documents found.")
            return

        for docID, sim in results:
            doc = self.df.loc[self.df["docID"] == docID]
            if not doc.empty:
                tweet_id = doc["tweetID"].values[0]
                text = doc["text"].values[0]
                print(f"TweetID: {tweet_id}\nText: {text}\nSimilarity: {sim:.4f}\n")

        save = input("Save result as CSV? (y/n): ").strip().lower()
        if save == "y":
            rows = []
            for docID, score in results:
                row = self.df[self.df["docID"] == docID]
                tweet_id = row.iloc[0]["tweetID"]
                text = row.iloc[0]["text"]
                rows.append({"tweet_id": tweet_id, "text": text, "score": score})
            pd.DataFrame(rows).to_csv("output/query_tfidf_result.csv", index=False)
            print("Results saved to query_tfidf_result.csv")
