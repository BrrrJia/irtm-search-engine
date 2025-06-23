import os
import joblib
from core.indexing import InvertedIndex
from core.classification import NaiveBayesClassifier
from sklearn.preprocessing import normalize
from core import config
from scipy.sparse import save_npz


# --- Helpers ---
def serialize_postings(postings_store):
    return {pid: plist.to_list() for pid, plist in postings_store.items()}


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

inv.postings_store = serialize_postings(inv.postings_store)

inv.build_tfidf_matrix()

save_npz("prebuilt/tfidf_matrix.npz", inv.tfidf_matrix)

inv.save()

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
data = normalize(inv_game.tfidf_matrix[: config.K_MEANS_DATA_SIZE], norm="l2")
save_npz("prebuilt/clustering_data.npz", data)

print("Clustering data saved.")
print("Preprocessing complete.")
