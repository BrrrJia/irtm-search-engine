import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
from PIL import Image
import io
from .core import config

# Page configuration
st.set_page_config(
    page_title="IRTM - Information Retrieval & Text Mining",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base configuration
API_BASE_URL = st.sidebar.text_input("API Base URL", config.API_BASE_URL)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #333;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🔍 IRTM - Information Retrieval & Text Mining System</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚀 Navigation")
selected_function = st.sidebar.selectbox(
    "Select Function",
    ["🔍 Document Search", "📚 Text Classification", "📊 Classification Model Evaluation", "🔬 Document Clustering"]
)

# Check API connection status
def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

# API connection status indicator
with st.sidebar:
    if check_api_connection():
        st.success("✅ API Connection Active")
    else:
        st.error("❌ API Connection Failed")
        st.warning("Please check API URL and service status")

# === Document Search Function ===
if selected_function == "🔍 Document Search":
    st.markdown('<div class="section-header">Document Search</div>', unsafe_allow_html=True)
    
    # Search configuration
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("🔍 Enter search query", placeholder="e.g., slee* cat, sleepy cat, side effects vaccines")
    
    with col2:
        search_mode = st.selectbox("Search Mode", ["term", "wildcard", "tfidf"])
    
    # Search mode explanation
    with st.expander("ℹ️ Search Mode Information"):
        st.write("**Term**: Boolean search with exact terms")
        st.write("**Wildcard**: Pattern matching with * (asterisk) wildcard")
        st.write("**TF-IDF**: Similarity-based ranking search")
    
    if st.button("🔍 Start Search", type="primary"):
        if query:
            with st.spinner("Searching documents..."):
                try:
                    params = {
                        "query": query,
                        "mode": search_mode
                    }
                    
                    start_time = time.time()
                    response = requests.get(f"{API_BASE_URL}/search", params=params)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        results = response.json()
                        search_time = end_time - start_time
                        
                        # Display search statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Results", results.get('total_count', 0))
                        with col2:
                            st.metric("Search Time", f"{search_time:.3f}s")
                        with col3:
                            st.metric("Search Mode", search_mode.upper())
                        with col4:
                            st.metric("Query Terms", len(query.split()))
                        
                        st.success(f"✅ Found {results.get('total_count', 0)} documents")
                        
                        # Display search results
                        if 'results' in results and results['results']:
                            st.markdown("### 📋 Search Results")
                            
                            for i, doc in enumerate(results['results'], 1):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <h4>📄 Document {i}: Tweet ID {doc.get('tweet_id', 'N/A')}</h4>
                                        <p>{doc.get('text', '')}</p>
                                        <small>🏷️ Tweet ID: {doc.get('tweet_id', 'N/A')}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                            # Create download data
                            df_results = pd.DataFrame(results['results'])
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("🔍 No documents found matching your query")
                    
                    elif response.status_code == 400:
                        st.error("❌ Invalid search parameters. Please check your query.")
                    elif response.status_code == 503:
                        st.error("❌ Search service is currently unavailable.")
                    else:
                        st.error(f"❌ Search failed with status code: {response.status_code}")
                        
                except requests.exceptions.RequestException:
                    st.error("❌ Connection error. Please check if the API is running.")
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a search query")

# === Text Classification Function ===
elif selected_function == "📚 Text Classification":
    st.markdown('<div class="section-header">Text Classification</div>', unsafe_allow_html=True)
    
    st.info("📝 This classifier predicts documents(German game reviews) sentiment: **gut** (good) or **schlecht** (bad)")
    
    # Classification input form
    with st.form("classification_form"):
        st.subheader("📝 Document Information")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", placeholder="Enter review name in German...")
        with col2:
            title = st.text_input("Title", placeholder="Enter review title in German...")
        
        review = st.text_area("Review Content", height=150,
                            placeholder="Enter review text in German...")
        
        submitted = st.form_submit_button("📚 Classify Document", type="primary")
    
    if submitted:
        if name.strip() or title.strip() or review.strip():
            with st.spinner("Classifying document..."):
                try:
                    data = {
                        "name": name,
                        "title": title,
                        "review": review
                    }
                    
                    response = requests.post(f"{API_BASE_URL}/classify", json=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display classification result
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            label = result.get('label', 'Unknown')
                            if label == 'gut':
                                st.success(f"🎯 Prediction: **{label}** (Good)")
                            elif label == 'schlecht':
                                st.error(f"🎯 Prediction: **{label}** (Bad)")
                            else:
                                st.info(f"🎯 Prediction: **{label}**")
                        
                        with col2:
                            combined_text = f"{name} {title} {review}".strip()
                            st.metric("Text Length", len(combined_text))
                        
                        with col3:
                            st.metric("Word Count", len(combined_text.split()))
                        
                        # Show input summary
                        with st.expander("📋 Input Summary"):
                            st.write(f"**Name**: {name}")
                            st.write(f"**Title**: {title}")
                            st.write(f"**Review**: {review[:200]}..." if len(review) > 200 else review)
                    
                    elif response.status_code == 400:
                        st.error("❌ Invalid input data. Please check your text.")
                    elif response.status_code == 503:
                        st.error("❌ Classification service is currently unavailable.")
                    else:
                        st.error(f"❌ Classification failed with status code: {response.status_code}")
                        
                except requests.exceptions.RequestException:
                    st.error("❌ Connection error. Please check if the API is running.")
                except Exception as e:
                    st.error(f"❌ Classification error: {str(e)}")
        else:
            st.warning("⚠️ Please provide at least one field (name, title, or review)")
    
    # Batch classification section
    st.markdown("---")
    st.subheader("📊 Batch Classification")
    
    uploaded_file = st.file_uploader("📁 Upload CSV file with columns: name, title, review", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📋 Preview of uploaded data:")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("📚 Classify All Documents", type="secondary"):
                if all(col in df.columns for col in ['name', 'title', 'review']):
                    with st.spinner("Processing batch classification..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, row in df.iterrows():
                            try:
                                data = {
                                    "name": str(row['name']) if pd.notna(row['name']) else "",
                                    "title": str(row['title']) if pd.notna(row['title']) else "",
                                    "review": str(row['review']) if pd.notna(row['review']) else ""
                                }
                                
                                response = requests.post(f"{API_BASE_URL}/classify", json=data)
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    results.append({
                                        'row_index': i,
                                        'name': data['name'][:50] + '...' if len(data['name']) > 50 else data['name'],
                                        'title': data['title'][:50] + '...' if len(data['title']) > 50 else data['title'],
                                        'prediction': result.get('label', 'Unknown'),
                                        'status': 'Success'
                                    })
                                else:
                                    results.append({
                                        'row_index': i,
                                        'name': data['name'][:50] + '...' if len(data['name']) > 50 else data['name'],
                                        'title': data['title'][:50] + '...' if len(data['title']) > 50 else data['title'],
                                        'prediction': 'Error',
                                        'status': 'Failed'
                                    })
                                
                                progress_bar.progress((i + 1) / len(df))
                                
                            except Exception as e:
                                results.append({
                                    'row_index': i,
                                    'name': 'Error',
                                    'title': 'Error', 
                                    'prediction': 'Error',
                                    'status': f'Exception: {str(e)}'
                                })
                        
                        # Display batch results
                        if results:
                            df_results = pd.DataFrame(results)
                            st.success(f"✅ Processed {len(results)} documents")
                            
                            # Classification distribution
                            prediction_counts = df_results['prediction'].value_counts()
                            fig_pie = px.pie(values=prediction_counts.values, names=prediction_counts.index,
                                           title='Classification Distribution')
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Results table
                            st.dataframe(df_results, use_container_width=True)
                            
                            # Download results
                            csv_results = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Classification Results",
                                data=csv_results,
                                file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("❌ CSV file must contain 'name', 'title', and 'review' columns")
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

# === Document Clustering Function ===
elif selected_function == "🔬 Document Clustering":
    st.markdown('<div class="section-header">Document Clustering Analysis</div>', unsafe_allow_html=True)
    
    st.info("📊 This function performs K-means clustering on document vectors and generates visualization")
    
    if st.button("🔬 Generate Clustering Visualization", type="primary"):
        with st.spinner("Performing clustering analysis..."):
            try:
                response = requests.get(f"{API_BASE_URL}/clustering")
                
                if response.status_code == 200:
                    # Display the clustering image
                    image = Image.open(io.BytesIO(response.content))
                    
                    st.success("✅ Clustering analysis completed successfully")
                    
                    # Display clustering visualization
                    st.image(image, caption="K-means Clustering Visualization", use_container_width=True)
                    
                    # Provide download option
                    st.download_button(
                        label="📥 Download Clustering Plot",
                        data=response.content,
                        file_name=f"clustering_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                    
                    # Additional information
                    with st.expander("ℹ️ Clustering Information"):
                        st.write("- **Algorithm**: K-means clustering")
                        st.write("- **Data**: Document TF-IDF vectors (first 200 documents)")
                        st.write("- **Highlighted**: Compact clusters with RSS values between 0 and 1")
                        st.write("- **Visualization**: 2D projection of document clusters")
                
                elif response.status_code == 503:
                    st.error("❌ Clustering service is currently unavailable. Data not prepared.")
                else:
                    st.error(f"❌ Clustering failed with status code: {response.status_code}")
                    
            except requests.exceptions.RequestException:
                st.error("❌ Connection error. Please check if the API is running.")
            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")

# === Model Evaluation Function ===
elif selected_function == "📊 Classification Model Evaluation":
    st.markdown('<div class="section-header">Classification Model Evaluation</div>', unsafe_allow_html=True)
    
    st.info("📈 Evaluate the trained Naive Bayes classifier performance using test dataset")
    
    if st.button("📊 Run Model Evaluation", type="primary"):
        with st.spinner("Evaluating classification model..."):
            try:
                response = requests.get(f"{API_BASE_URL}/classify/evaluate")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Model evaluation completed successfully")
                    
                    # Display evaluation metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        accuracy = float(result.get('Accuracy', '0'))
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    with col2:
                        f1_score = float(result.get('F1', '0'))
                        st.metric("F1 Score", f"{f1_score:.4f}")
                    
                    with col3:
                        timestamp = result.get('timestamp', 'N/A')
                        st.write("**Evaluation Time**")
                        st.write(timestamp.split('T')[0] if 'T' in timestamp else timestamp)
                    
                    # Metric interpretation
                    st.markdown("### 📊 Performance Interpretation")
                    
                    performance_level = "Excellent" if accuracy > 0.9 else "Good" if accuracy > 0.8 else "Fair" if accuracy > 0.7 else "Poor"
                    
                    if accuracy > 0.8:
                        st.success(f"🎯 **{performance_level}** classification performance (Accuracy: {accuracy:.1%})")
                    elif accuracy > 0.6:
                        st.warning(f"⚠️ **{performance_level}** classification performance (Accuracy: {accuracy:.1%})")
                    else:
                        st.error(f"❌ **{performance_level}** classification performance (Accuracy: {accuracy:.1%})")
                    
                    
                    # Performance guidelines
                    with st.expander("📋 Metric Definitions"):
                        st.write("**Accuracy**: Proportion of correct predictions among total predictions")
                        st.write("**F1 Score**: Harmonic mean of precision and recall")
                        st.write("**Performance Levels**:")
                        st.write("- Excellent: > 90%")
                        st.write("- Good: 80-90%") 
                        st.write("- Fair: 70-80%")
                        st.write("- Poor: < 70%")
                
                elif response.status_code == 503:
                    st.error("❌ Classification service is currently unavailable. Cannot perform evaluation.")
                else:
                    st.error(f"❌ Evaluation failed with status code: {response.status_code}")
                    
            except requests.exceptions.RequestException:
                st.error("❌ Connection error. Please check if the API is running.")
            except Exception as e:
                st.error(f"❌ Evaluation error: {str(e)}")
    
    # Model information section
    st.markdown("---")
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Algorithm**: Naive Bayes Classifier
        
        **Task**: German game review sentiment classification
        
        **Classes**: 
        - gut (good)
        - schlecht (bad)
        """)
    
    with col2:
        st.info("""
        **Features**: 
        - Review name
        - Review title  
        - Review content
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    🔍 IRTM System - Information Retrieval & Text Mining<br>
    Built with Streamlit & FastAPI | 📧 Contact: youjiaim@protonmail.com
</div>
""", unsafe_allow_html=True)