# IRTM Engine

**IRTM Engine** is an information retrieval and text mining system that supports:

- **Query-based search** on a bilingual tweet corpus
- **Text classification and clustering** on a German game review dataset

It combines classical IR techniques(inverted indexing, boolean query, TF-IDF) with basic machine learning methods such as Naive Bayes classification and K-means clustering.

## System Architecture

The system consists of three core pipelines:

### 1. Document Indexing

- Tokenization and text preprocessing
- Inverted index construction with support for term, bigram, and permuterm indexing
- TF-IDF matrix generation for document-term vector representation

### 2. Document Retrieval

- Boolean query engine with support for exact term and wildcard(\*) matching
- Similarity-based retrieval using TF-IDF and cosine similarity

### 3. Text Analytics

- **Classification**: Supervised sentiment classification using a Naive Bayes model trained on TF-IDF features
- **Evaluation**: Model performance evaluation using Accuracy and F1-score
- **Clustering**: Unsupervised document clustering using K-means and optimal K-means based on TF-IDF vectors

## How to Run

### Option 1: Run from Terminal (CLI Mode)

This mode allows you to interact with the IRTM engine via the command line for query search, classification, and clustering.

1. Clone this repository

```bash
git clone https://github.com/BrrrJia/irtm-search-engine.git
```

2. Set up a Python environment and install dependencies

```bash
cd backend
python -m venv irtm-env
# Activate the virtual environment
source irtm-env/bin/activate      # On macOS/Linux
# .\irtm-env\Scripts\activate     # On Windows

# Install required packages
pip install -r requirements.txt
```

3. Choose a mode to run:

- Interactive CLI (menu-driven):

```bash
python -m cli.main
```

- Direct command (argparse-based):

```bash
python -m cli.main --task query --query_type wildcard --query "cancer detect*" # boolean wildcard query mode
python -m cli.main --task classify # classification task
python -m cli.main --task cluster # clustering task
```

### Option 2: Run with Docker

This mode runs both the backend API (FastAPI) and frontend UI (Streamlit) in isolated containers.

1. Clone this repository

```markdown
git clone https://github.com/BrrrJia/irtm-search-engine.git
```

2. Build and start the services

> Make sure Docker is running **before** executing the following command.

You can check with `docker info`. If not started, launch Docker Desktop first.

```bash
docker compose -f docker-compose.yml up --build
```

This will:

- Build the backend (API) and frontend (Streamlit) services
- Expose:
  - FastAPI at `http://localhost:8000`
  - Streamlit at `http://localhost:8501`

### Option 3: Hosted on Render

The app is deployed and publicly accessible via Render.

Visit: [https://irtm-ui.onrender.com](https://irtm-ui.onrender.com/)

> Note: Render services may take up to 3-4 minutes to cold-start.

### Troubleshooting

If the UI shows “API not available”:

- The backend may still be warming up
- Refresh the page after 30 seconds
- Alternatively, check `/health` endpoints on the backend manually

## Data

- **tweets.csv**
  - Total: 120,428 entries
  - 7,000 entries used in the query search task (adjustable via config)
  - Source: Provided by the lecturer
- **games-train.csv**
  - 124,063 entries used for classification
  - 200 entries used for clustering
- **games-test.csv**
  - 44,233 entries used for classification evaluation

> Both `games-train.csv` and `games-test.csv` are derived from the [LREC 2016 paper](http://www.lrec-conf.org/proceedings/lrec2016/pdf/59_Paper.pdf).

## Python Packages

Core packages used in this project include:

- `nltk`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `scipy`
- `fastapi`
- `uvicorn`
- `streamlit`
- `joblib`
- `requests`

## License & Credits

This project is for educational use. Dataset partially provided by the course and public research resources.
