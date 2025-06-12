# IRTM Engine

A lightweight **Information Retrieval and Text Mining engine** supporting multilingual tweet search and German game review classification/clustering. Designed for NLP coursework and deployable via **CLI**, **FastAPI**, and **Docker**.

---

## 🚀 Features

- 🔎 **Information Retrieval (tweets.csv)**

  - Boolean Term Queries (AND)
  - Boolean Wildcard Queries (Bigram / Permuterm)
  - TF-IDF Similarity Ranking

- 🧠 **Document Classification (game data)**

  - Naive Bayes text classifier for German app reviews

- 📊 **Document Clustering**

  - K-Means and Optimal KMeans with PCA + visualization

- 🖥️ **Modes of Operation**

  - Command Line Interface (CLI)
  - FastAPI-based Web API
  - [Planned] Streamlit UI frontend + Render deployment

- 🐳 **DevOps & Config**
  - Modular project layout (`app/indexing.py`, `routes/query.py`, etc.)
  - Configurable via `.env` file
  - Docker & docker-compose ready
  - [TODO] GitHub Actions for testing

---

## 🗂️ Project Structure

```
irtm-search-engine/
├── data/                     # Datasets: tweets and game reviews
│   ├── tweets.csv
│   ├── games-train.csv
│   └── games-test.csv
├── output/                   # Output results: TF-IDF, clustering, etc.
│   ├── query_boolean_result.csv
│   ├── query_tfidf_result.csv
│   └── k-means.png
├── docker/                   # Docker-related configs
│   ├── Dockerfile.api
│   └── docker-compose.yml
├── src/                      # Source code base
│   ├── api/                  # FastAPI backend
│   │   ├── main.py
│   │   ├── exceptions.py
│   │   ├── models.py
│   │   └── routes/           # API endpoints
│   │       ├── search.py
│   │       ├── classify.py
│   │       ├── clustering.py
│   │       └── evaluate.py
│   ├── cli/                  # CLI interface
│   │   └── main.py
│   |── core/                 # Core logic modules
│   |   ├── indexing.py
│   |   ├── retrieval.py
│   |   ├── classification.py
│   |   ├── clustering.py
│   |   ├── utils.py
│   |   └── config.py
|   └── streamlit_app.py      # Streamlit frontend
├── tests/                    # Pytest test files
│   └── test_api.py
├── requirements.txt          # Python dependencies
├── Makefile                  # Common commands (build/run/test)
├── pytest.ini                # Pytest config
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Usage

### 🧪 Local CLI (Interactive Mode)

Run the CLI tool interactively:

```bash
python -m src.cli.main
```

You will be prompted to choose a task:
→ `query`, `classify`, or `cluster`
If `query` is selected, you can choose the query type (`bool`, `tfidf`, `wildcard`) and input terms accordingly.

### 🧪 Local CLI (Argument Mode)

You can also run CLI tasks non-interactively using command-line arguments:

```bash
# Run a wildcard query
python -m src.cli.main --task query --query_type wildcard --query "slee* cat"

# Run classification pipeline (train → test → evaluate)
python -m src.cli.main --task classify

# Run KMeans clustering and save visualization
python -m src.cli.main --task cluster
```

### 🖥️ API Server

```bash
uvicorn src.api.main:app --reload --port=8000
```

### 🐳 Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## 📂 Dataset Info

- **tweets.csv**

  - Small English/German tweet dataset (provided by instructor)
  - Used for retrieval experiments

- **games-train.csv / games-test.csv**
  - German mobile app reviews (from LREC 2016 dataset)
  - Used for classification and clustering
  - Source: [LREC paper](http://www.lrec-conf.org/proceedings/lrec2016/pdf/59_Paper.pdf)

---

## 📊 Visualization Output

KMeans clustering on PCA-reduced vectors (TF-IDF), visualized and saved to `output/k-means.png`

---

## 🌐 Deployment [Planned]

- Backend: FastAPI on Render
- Frontend: Streamlit or HTML+JS
- Environment config via `.env`
- API example routes:
  - `GET /query?type=tfidf&text=...`
  - `POST /classify`

---

## 📌 Tech Stack

- Python 3.11
- FastAPI / Uvicorn
- Docker / docker-compose
- Numpy / Pandas / Scikit-learn / Matplotlib
- Streamlit [planned]

---

## 🙏 Credits

Project developed as part of the "Information Retrieval and Text Mining" course @ University of Stuttgart.

German app review data used under academic license from LREC 2016 dataset.

---
