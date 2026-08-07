import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import pickle
import time
from collections import Counter
from huggingface_hub import hf_hub_download, HfApi

# Add current folder to python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom architectures and utilities from src/models
try:
    from models.scratch import build_model
    from src.scratch_preprocess import clean_text, encode_text, MAX_LEN as SCRATCH_MAX_LEN
except Exception as e:
    st.warning(f"Failed to import deep learning utilities: {e}")

# Page config
st.set_page_config(
    page_title="Smart MCQ Solver & Confidence Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.sub-title {
    color: #94A3B8;
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 2rem;
}

.card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    margin-bottom: 1rem;
    transition: all 0.25s ease;
}

.card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-2px);
}

.result-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
    border: 2px solid #6366F1;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
    margin-bottom: 1.5rem;
}

.answer-badge {
    background: #6366F1;
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    border-radius: 10px;
    padding: 0.3rem 1.5rem;
    display: inline-block;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}

.answer-text {
    font-size: 1.25rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-top: 0.5rem;
}

.metric-badge {
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

.badge-green {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-red {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-amber {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Sidebar - Diagnostics and Status
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Dashboard Control</h2>", unsafe_allow_html=True)

# Hardware Detection
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else ""

st.sidebar.markdown("### 🖥️ Hardware Status")
if device == "cuda":
    st.sidebar.markdown(f'<div class="metric-badge badge-green">🟢 GPU: {gpu_name}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="metric-badge badge-amber">🟡 Running on CPU (No CUDA)</div>', unsafe_allow_html=True)

# Hugging Face Configuration
st.sidebar.markdown("### 🌐 Hugging Face Integration")
repo_id = st.sidebar.text_input("Hugging Face Repository ID", value="jigyasa07/smart_mcq_solver")
hf_token = st.sidebar.text_input("HF Access Token (optional)", value=os.environ.get("HF_TOKEN", ""), type="password")

@st.cache_resource
def check_hf_files(repo_id, token=None):
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, token=token)
        return {
            'tfidf': 'tfidf.pkl' in files,
            'scratch_model': 'model.pt' in files,
            'scratch_vocab': 'word2idx.pkl' in files,
            'error': None
        }
    except Exception as e:
        return {
            'tfidf': False,
            'scratch_model': False,
            'scratch_vocab': False,
            'error': str(e)
        }

# Model Artifacts Status
st.sidebar.markdown("### 📦 Hugging Face Registry")
if repo_id:
    hf_status = check_hf_files(repo_id, token=hf_token if hf_token else None)
    if hf_status['error']:
        st.sidebar.markdown(f'<div class="metric-badge badge-red">✗ Error connecting: {hf_status["error"][:50]}...</div>', unsafe_allow_html=True)
    else:
        if hf_status['tfidf']:
            st.sidebar.markdown('<div class="metric-badge badge-green">✓ tfidf.pkl found</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="metric-badge badge-red">✗ tfidf.pkl missing</div>', unsafe_allow_html=True)

        if hf_status['scratch_model'] and hf_status['scratch_vocab']:
            st.sidebar.markdown('<div class="metric-badge badge-green">✓ DL Scratch files found</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="metric-badge badge-red">✗ DL Scratch files missing</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="metric-badge badge-amber">⚠ Please enter HF Repo ID</div>', unsafe_allow_html=True)

# Select Model
st.sidebar.markdown("### 🧠 Select Active Model")
selected_model_name = st.sidebar.selectbox(
    "Choose AI Solver",
    ["TF-IDF Classifier (Ultra Fast)", 
     "Custom Deep Learning (BiLSTM + CNN + Attention)"]
)

# ----------------------------------------------------
# Cache model loadings from HF
# ----------------------------------------------------

@st.cache_resource
def load_tfidf_from_hf(repo_id, token=None):
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="tfidf.pkl",
            token=token
        )
        bundle = joblib.load(model_path)
        return bundle['vectorizer'], bundle['model']
    except Exception as e:
        st.error(f"Error loading TF-IDF from Hugging Face: {e}")
        return None, None

@st.cache_resource
def load_scratch_from_hf(repo_id, _device, token=None):
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.pt",
            token=token
        )
        vocab_path = hf_hub_download(
            repo_id=repo_id,
            filename="word2idx.pkl",
            token=token
        )
        with open(vocab_path, 'rb') as f:
            word2idx = pickle.load(f)
            
        checkpoint = torch.load(model_path, map_location=_device)
        vocab_size = checkpoint['vocab_size']
        model = build_model(vocab_size).to(_device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return word2idx, model
    except Exception as e:
        st.error(f"Error loading Deep Learning model from Hugging Face: {e}")
        return None, None

# Main Page Design
st.markdown('<div class="main-title">🧠 MCQ Smart Solver Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deploying Deep Learning and NLP Models from Hugging Face to Resolve Multiple-Choice Questions</div>', unsafe_allow_html=True)

# Initialize loaded models based on selection
active_vectorizer, active_tfidf_model = None, None
active_vocab, active_dl_model = None, None

hf_token_passed = hf_token if hf_token else None

if "TF-IDF" in selected_model_name:
    if repo_id:
        with st.spinner("Loading TF-IDF vectorizer and model from Hugging Face..."):
            active_vectorizer, active_tfidf_model = load_tfidf_from_hf(repo_id, token=hf_token_passed)
        if active_tfidf_model is not None:
            st.success("⚡ TF-IDF solver loaded successfully from Hugging Face!")
    else:
        st.error("❌ Please provide a valid Hugging Face Repository ID in the sidebar.")

elif "Custom Deep Learning" in selected_model_name:
    if repo_id:
        with st.spinner("Loading BiLSTM + CNN + Attention model architecture from Hugging Face..."):
            active_vocab, active_dl_model = load_scratch_from_hf(repo_id, device, token=hf_token_passed)
        if active_dl_model is not None:
            st.success("🤖 Custom Deep Learning solver loaded successfully from Hugging Face!")
    else:
        st.error("❌ Please provide a valid Hugging Face Repository ID in the sidebar.")

# Tabs
tab1, tab2 = st.tabs(["🎯 Solve Single Question", "⚙️ Model Explanations"])

with tab1:
    st.markdown("### Input Multiple Choice Question")
    
    col_q, col_opts = st.columns([1.2, 1])
    
    with col_q:
        prompt_input = st.text_area(
            "Question/Prompt Text",
            placeholder="Type your question prompt here...",
            height=180,
            value="Which of the following is a key component of a convolutional neural network?"
        )
    
    with col_opts:
        st.write("Options")
        opt_a = st.text_input("Option A", value="Max Pooling Layer")
        opt_b = st.text_input("Option B", value="Recurrent Neural Unit")
        opt_c = st.text_input("Option C", value="Decision Boundary")
        opt_d = st.text_input("Option D", value="Support Vector Classifier")
        opt_e = st.text_input("Option E", value="Bag of Words model")
        
    options_dict = {'A': opt_a, 'B': opt_b, 'C': opt_c, 'D': opt_d, 'E': opt_e}
    
    # Process Single Inference
    if st.button("🚀 Analyze MCQ", use_container_width=True):
        if not prompt_input.strip() or any(not str(v).strip() for v in options_dict.values()):
            st.error("Please fill in the question prompt and all five options.")
        else:
            scores = []
            
            with st.spinner("AI Solver is thinking..."):
                # TF-IDF Inference
                if "TF-IDF" in selected_model_name and active_tfidf_model is not None:
                    rows = []
                    for opt in ['A', 'B', 'C', 'D', 'E']:
                        text = prompt_input + ' [SEP] ' + str(options_dict[opt])
                        rows.append({'text': text})
                    df_temp = pd.DataFrame(rows)
                    X_tfidf = active_vectorizer.transform(df_temp['text'])
                    # proba of correct class (1)
                    proba = active_tfidf_model.predict_proba(X_tfidf)[:, 1]
                    scores = proba.tolist()
                
                # Custom DL Inference
                elif "Custom Deep Learning" in selected_model_name and active_dl_model is not None:
                    cleaned_prompt = clean_text(prompt_input)
                    seqs = []
                    for opt in ['A', 'B', 'C', 'D', 'E']:
                        option_text = clean_text(options_dict[opt])
                        text = f"question {cleaned_prompt} answer {option_text}"
                        seq = encode_text(text, active_vocab, SCRATCH_MAX_LEN)
                        seqs.append(seq)
                    tensor = torch.tensor(seqs, dtype=torch.long).to(device)
                    with torch.no_grad():
                        logits = active_dl_model(tensor)
                        probs = torch.sigmoid(logits).cpu().numpy()
                    scores = probs.tolist()
                else:
                    st.error("Selected model is not loaded or missing weights. Please verify the active model settings.")
            
            if len(scores) == 5:
                # Softmax or Normalize scores to sum to 100% for relative confidence comparison
                score_sum = sum(scores)
                normalized_conf = [s / score_sum * 100 for s in scores] if score_sum > 0 else [20.0] * 5
                
                # Rank results
                results_df = pd.DataFrame({
                    'Option': ['A', 'B', 'C', 'D', 'E'],
                    'Content': [opt_a, opt_b, opt_c, opt_d, opt_e],
                    'Score': scores,
                    'Confidence (%)': normalized_conf
                })
                
                sorted_results = results_df.sort_values(by='Confidence (%)', ascending=False)
                top_option = sorted_results.iloc[0]['Option']
                top_text = sorted_results.iloc[0]['Content']
                
                # Output UI Design
                st.write("---")
                col_res, col_chart = st.columns([1, 1.2])
                
                with col_res:
                    st.markdown("#### 🏆 Top Prediction")
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="answer-badge">{top_option}</div>
                        <div class="answer-text">"{top_text}"</div>
                        <div style="font-size:1rem; color:#A855F7; font-weight:600; margin-top:1rem;">
                            Model Confidence: {sorted_results.iloc[0]['Confidence (%)']:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 3 ranked options space separated (Kaggle submission format)
                    ranked_list = sorted_results['Option'].tolist()
                    submission_format = " ".join(ranked_list[:3])
                    st.info(f"💾 **Submission Prediction String (Top 3):** `{submission_format}`")
                
                with col_chart:
                    st.markdown("#### 📊 Option Confidence Distribution")
                    chart_data = pd.DataFrame({
                        'Confidence (%)': normalized_conf
                    }, index=['A', 'B', 'C', 'D', 'E'])
                    
                    st.bar_chart(chart_data, color="#6366F1")
                    
                    # Small details table
                    st.dataframe(
                        results_df.rename(columns={'Score': 'Raw Output'}),
                        use_container_width=True,
                        hide_index=True
                    )

with tab2:
    st.markdown("### 🔍 Model Architecture Explanations")
    
    st.markdown("""
    This project evaluates multiple approaches to solve multiple choice questions under a common framework, building and comparing features across two models:
    
    1. **TF-IDF Baseline Classifier**
       - **How it works:** Extends each question into 5 instances, concatenating prompt and options with a special `[SEP]` token. TF-IDF extracts word-level features which are mapped onto a logistic/SGD classifier to predict option probability.
       - **Best used for:** Low resource inference, instantaneous prediction, and establishing baseline performance.
       - **Hugging Face file name:** `tfidf.pkl`
       
    2. **Custom Deep Learning Network (BiLSTM + CNN + Attention)**
       - **How it works:** Implements a word-embedded model starting with custom 256-dimensional embeddings. It processes features using a 1D Convolutional layer to capture local patterns, runs sequence analysis via a Bidirectional LSTM (BiLSTM), and weights importances dynamically via a custom soft Attention mechanism.
       - **Best used for:** Mid-tier architectures where sequence dependencies and token context matter, without importing heavy transformer pre-train models.
       - **Hugging Face file names:** `model.pt` and `word2idx.pkl`
    """)

st.divider()
st.caption("©Jigyasa- Smart MCQ Solver Dashboard. All rights reserved.")
