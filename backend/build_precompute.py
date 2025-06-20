import os
import numpy as np
import joblib
from core.indexing import InvertedIndex
from core.classification import NaiveBayesClassifier
from sklearn.preprocessing import normalize
from core import config


# --- Helpers ---
def serialize_postings(postings_store):
    return {
        pid: plist.to_list() for pid, plist in postings_store.items()
    }

# create folder for pre-built data
os.makedirs("prebuilt", exist_ok=True)

print("Building inverted index and TF-IDF matrix...")

# === Inverted Index + TF-IDF ===
inv = InvertedIndex(config.DATA_PATH)
inv.index()
if config.USE_BIGRAM:
    inv.bigram_index()
if config.USE_PERMUTERM:
    inv.permuterm_index()
inv.build_tfidf_matrix()

joblib.dump({
    "term_dictionary": inv.term_dictionary,
    "postings_store":  serialize_postings(inv.postings_store),
    "df": inv.df,
    "bigram_dictionary": inv.bigram_dictionary,
    "permuterm_dictionary": inv.permuterm_dictionary,
    "term_to_index": inv.term_to_index,
    "idf_vector": inv.idf_vector,
}, "prebuilt/inverted_index_dict.pkl")
np.save("prebuilt/tfidf_matrix.npy", inv.tfidf_matrix)

print("Inverted index saved.")

# === Classifier ===
print("Training classifier...")
cls = NaiveBayesClassifier(config.TRAIN_PATH)
cls.train()
joblib.dump(cls, "prebuilt/classifier.pkl")
print("Classifier saved.")

# === Clustering data ===
print("Preparing clustering data...")
inv_game = InvertedIndex(config.TRAIN_PATH)
inv_game.index()
inv_game.build_tfidf_matrix()
data = normalize(inv_game.tfidf_matrix[:config.K_MEANS_DATA_SIZE], norm="l2")
np.save("prebuilt/clustering_data.npy", data)

print("Clustering data saved.")
print("Preprocessing complete.")