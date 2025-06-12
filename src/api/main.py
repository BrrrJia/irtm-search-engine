from ..core.indexing import InvertedIndex
from ..core.retrieval import RetrievalEngine
from ..core import config
from ..core.classification import NaiveBayesClassifier
from .routes import search, classify, evaluate, clustering
from fastapi import FastAPI
import logging
from sklearn.preprocessing import normalize


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
]

app = FastAPI(
    title="IRTM API",
    description="Information Retrieval and Text Minning API with query search, text classification, and clustering capabilities",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.include_router(search.router)
app.include_router(classify.router)
app.include_router(evaluate.router)
app.include_router(clustering.router)

# === Initialisation ===
inv = None
ret = None
cls = None
data = None


@app.on_event("startup")
async def initialize_components():
    try:
        logger.info("Initializing inverted index...")
        inv = InvertedIndex(config.DATA_PATH)
        inv.index()
        if config.USE_BIGRAM:
            inv.bigram_index()
        if config.USE_PERMUTERM:
            inv.permuterm_index()

        inv.build_tfidf_matrix()
        logger.info("Inverted index initialized successfully")

        logger.info("Initializing retrieval engine...")
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
        logger.info("Retrieval engine initialized successfully")

        app.state.inv = inv
        app.state.ret = ret

    except Exception as e:
        logger.error(f"Failed to initialize search components: {str(e)}")
        app.state.inv = None
        app.state.ret = None

    try:
        logger.info("Initializing classifier...")
        cls = NaiveBayesClassifier(config.TRAIN_PATH)
        cls.train()
        logger.info("Classifier trained successfully")

        app.state.cls = cls

    except Exception as e:
        logger.error(f"Failed to initialize classifier: {str(e)}")
        app.state.cls = None

    try:
        logger.info("Preparing clustering data...")
        inv_game = InvertedIndex(config.TRAIN_PATH)
        inv_game.index()
        inv_game.build_tfidf_matrix()
        data = normalize(inv_game.tfidf_matrix[:config.K_MEANS_DATA_SIZE], norm="l2")
        logger.info("Clustering data prepared successfully")

        app.state.data = data

    except Exception as e:
        logger.error(f"Failed to prepare clustering data: {str(e)}")
        app.state.data = None
