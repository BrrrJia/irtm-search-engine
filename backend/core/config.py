import os
from dotenv import load_dotenv

load_dotenv()

INDEX_DATA_SIZE = int(os.getenv("INDEX_DATA_SIZE", 7000))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # core/
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "tweets.csv")

USE_BIGRAM = os.getenv("USE_BIGRAM", "True") == "True"
USE_PERMUTERM = os.getenv("USE_PERMUTERM", "False") == "True"
TOP_K = int(os.getenv("TOP_K", 10))

TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "games-train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "games-test.csv")

K_MEANS_DATA_SIZE = int(os.getenv("K_MEANS_DATA_SIZE", 200))
USE_OPTIMAL_KMEANS = os.getenv("USE_OPTIMAL_KMEANS", "False") == "True"
K_MEANS_K = int(os.getenv("K_MEANS_K", 20))
K_MEANS_N_INIT = int(os.getenv("K_MEANS_N_INIT", 50))