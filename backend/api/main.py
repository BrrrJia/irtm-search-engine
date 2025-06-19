from core.indexing import InvertedIndex
from core.retrieval import RetrievalEngine
from core import config
from core.classification import NaiveBayesClassifier
from api.routes import search, classify, evaluate, clustering
from fastapi import FastAPI
import logging
from sklearn.preprocessing import normalize
import os
import uvicorn
import joblib
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define tags metadata
tags_metadata = [
    {
        "name": "search",
        "description": "Query search supporting for boolean term/wildcard(*), and TF-IDF search.",
    },
    {
        "name": "classification",
        "description": "Text classification using Naive Bayes classifier.",
    },
    {
        "name": "clustering",
        "description": "Document clustering analysis using K-means algorithm with visualization.",
    },
    {
        "name": "status",
        "description": "Health and component readiness check for the API.",
    },
]

app = FastAPI(
    title="IRTM API",
    description="Information Retrieval and Text Minning API with query search, text classification, and clustering capabilities",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

@app.get("/", tags=["status"])
def root():
    return {
        "project": "IRTM API",
        "status": "API is running",
        "documentation": "/docs",
        "available_endpoints": [
            "/search",
            "/classify",
            "/evaluate",
            "/clustering",
            "/status",
            "/health"
        ]
    }

@app.get("/health", tags=["status"])
def health():
    return {"status": "ok"}


@app.get("/status", tags=["status"])
def check_status():
    return {
        "inverted_index_loaded": app.state.inv is not None,
        "retrieval_engine_loaded": app.state.ret is not None,
        "classifier_loaded": app.state.cls is not None,
        "clustering_data_loaded": app.state.data is not None
    }


# register API routes
app.include_router(search.router)
app.include_router(classify.router)
app.include_router(evaluate.router)
app.include_router(clustering.router)

# initialisation
@app.on_event("startup")
async def initialize_components():
    # === Inverted Index + TF-IDF ===
    try:
        logger.info("Loading inverted index from prebuilt cache...")
        inv_dict = joblib.load("prebuilt/inverted_index_dict.pkl")
        tfidf_matrix = np.load("prebuilt/tfidf_matrix.npy")

        ret = RetrievalEngine(
            inv_dict["term_dictionary"],
            inv_dict["postings_store"],
            inv_dict["df"],
            inv_dict["bigram_dictionary"],
            inv_dict["permuterm_dictionary"],
            inv_dict["term_to_index"],
            inv_dict["idf_vector"],
            tfidf_matrix
        )
        app.state.inv = inv_dict
        app.state.ret = ret
        logger.info("Inverted index & retrieval engine loaded from cache.")
    except Exception as e:
        logger.warning(f"Failed to load cache: {e} — rebuilding index...")
        try:
            inv = InvertedIndex(config.DATA_PATH)
            inv.index()
            if config.USE_BIGRAM:
                inv.bigram_index()
            if config.USE_PERMUTERM:
                inv.permuterm_index()
            inv.build_tfidf_matrix()

            ret = RetrievalEngine(
                inv.term_dictionary,
                inv.postings_store,
                inv.df,
                inv.bigram_dictionary,
                inv.permuterm_dictionary,
                inv.term_to_index,
                inv.idf_vector,
                inv.tfidf_matrix,
            )
            app.state.inv = inv
            app.state.ret = ret
            logger.info("Inverted index rebuilt at runtime.")
        except Exception as e:
            logger.error(f"Failed to initialize search components: {str(e)}")
            app.state.inv = None
            app.state.ret = None

    # === Classifier ===
    try:
        logger.info("Loading classifier from cache...")
        cls = joblib.load("prebuilt/classifier.pkl")
        app.state.cls = cls
        logger.info("Classifier loaded from cache.")
    except Exception as e:
        logger.warning(f"Failed to load classifier: {e} — retraining...")
        try:
            cls = NaiveBayesClassifier(config.TRAIN_PATH)
            cls.train()
            app.state.cls = cls
            logger.info("Classifier retrained at runtime.")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {str(e)}")
            app.state.cls = None

    # === Clustering data ===
    try:
        logger.info("Loading clustering data from cache...")
        data = np.load("prebuilt/clustering_data.npy")
        app.state.data = data
        logger.info("Clustering data loaded.")
    except Exception as e:
        logger.warning(f"Failed to load clustering data: {e} — rebuilding...")
        try:
            inv_game = InvertedIndex(config.TRAIN_PATH)
            inv_game.index()
            inv_game.build_tfidf_matrix()
            data = normalize(inv_game.tfidf_matrix[:config.K_MEANS_DATA_SIZE], norm="l2")
            app.state.data = data
            logger.info("Clustering data rebuilt at runtime.")
        except Exception as e:
            logger.error(f"Failed to prepare clustering data: {str(e)}")
            app.state.data = None


if __name__ == "__main__":
    # Automatically detect Render's $PORT environment variable, otherwise use the default of 8000.
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run("api.main:app", host="0.0.0.0", port=port)