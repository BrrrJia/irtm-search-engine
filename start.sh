# Start FastAPI on port 8000
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit on port 8501
streamlit run src/streamlit_app.py --server.port=8501 --server.enableCORS=false