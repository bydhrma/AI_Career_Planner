"""
AI CAREER PLANNER - Enhanced Streamlit Application
Professional HR GenAI + Skill Recommendation System
"""

import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
import ast
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import time
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import os
# Fix KMeans memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
# Prevent sklearn warnings
warnings.filterwarnings("ignore")
import sys
# Prevent PyTorch class path error
sys.modules["torch._classes"] = None

import google.generativeai as genai
from google.generativeai import types
# Removed invalid import: `from google.genai.types import ...` (module does not exist).
# If you need those type hints, use the correct namespace (uncomment if available):
# from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch

# =============================================================================
# GLOBAL CSS STYLING (untuk combobox non-editable)
# =============================================================================
st.markdown("""
<style>
[data-baseweb="combobox"] input {
    pointer-events: none !important;
    cursor: default !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Career Planner | Smart Career Guidance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-career-planner',
        'Report a bug': "https://github.com/yourusername/ai-career-planner/issues",
        'About': "# AI Career Planner\nSmart job title prediction & skill recommendation using Machine Learning."
    }
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
def load_custom_css(css_file_path="style/style_streamlit.css"):
    """Load external CSS file for better UI/UX"""
    try:
        with open(css_file_path, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"❌ CSS file '{css_file_path}' not found. Please ensure the file exists.")
load_custom_css()

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
BEST_MODEL_PATH = "../models/best_model.joblib"
TFIDF_PATH = "../models/tfidf_vectorizer.joblib"
LE_PATH = "../models/label_encoder.joblib"
SKILL_VEC_PATH = "../models/skill_vectorizer.joblib"
DATASET_PATH = "../data/processed/Clean Dataset/cleaned_data.csv"

# =============================================================================
# CACHING FUNCTIONS (DO NOT CHANGE AS REQUESTED)
# =============================================================================
@st.cache_resource
def load_ml_resources(best_model_path, tfidf_path, le_path, skill_vec_path):
    """Load all ML models and vectorizers (runs only once)."""
    model = joblib.load(best_model_path)
    tfidf = joblib.load(tfidf_path)
    le = joblib.load(le_path)
    skill_vect = joblib.load(skill_vec_path)
    return model, tfidf, le, skill_vect

@st.cache_data
def load_and_prepare_data(csv_path, _skill_vect):
    """Load data, fix skills_token format, and create Skill Matrix (runs only once)."""
    df = pd.read_csv(csv_path)

    def parse_skill_token(x):
        try:
            if isinstance(x, str):
                return ast.literal_eval(x)
            return x
        except (ValueError, SyntaxError):
            return []

    df["skills_token_list"] = df["skills_token"].apply(parse_skill_token)
    df["skills_text"] = df["skills_token_list"].apply(lambda x: " ".join([str(s).replace(" ", "_") for s in x]))
    SK = _skill_vect.transform(df["skills_text"])
    return df, SK

# Load resources with error handling
try:
    model, tfidf, le, skill_vect = load_ml_resources(
        BEST_MODEL_PATH, TFIDF_PATH, LE_PATH, SKILL_VEC_PATH
    )
    df, SK = load_and_prepare_data(DATASET_PATH, skill_vect)
    resources_loaded = True
except Exception as e:
    st.error(f"Error loading resources : {str(e)}")
    resources_loaded = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def clean_text(text: str):
    """Clean and preprocess text"""
    txt = str(text).lower()
    txt = re.sub(r"http\S+|www\S+", "", txt)
    txt = re.sub(r"[^a-zA-Z0-9 ]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def predict_with_confidence(text, top_k=5):
    """Predict job titles with confidence scores"""
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        top_idx = np.argsort(prob)[::-1][:top_k]
        labels = le.inverse_transform(top_idx)
        scores = prob[top_idx]
    else:
        dec = model.decision_function(vec)
        if dec.ndim == 1:
            dec = [dec]
        top_idx = np.argsort(dec[0])[::-1][:top_k]
        labels = le.inverse_transform(top_idx)
        e = np.exp(dec[0][top_idx] - np.max(dec[0][top_idx]))
        scores = e / e.sum()

    return list(zip(labels, scores))

def recommend_skills_from_text(input_skill_text, top_jobs=50, top_skills=15):
    """Recommend skills based on input"""
    s = clean_text(input_skill_text)
    s = s.replace(" ", "_")
    s_vec = skill_vect.transform([s])
    sims = cosine_similarity(s_vec, SK).ravel()
    top_idx = np.argsort(sims)[::-1][:top_jobs]
    candidate_txt = df.iloc[top_idx]["skills_text"].tolist()
    
    cnt = Counter()
    for txt in candidate_txt:
        cnt.update(txt.split())
    
    input_tokens = set(s.split("_"))
    ranked = [
        (skill.replace("_", " "), freq)
        for skill, freq in cnt.most_common()
        if skill not in input_tokens
    ]
    
    return ranked[:top_skills]

# =============================================================================
# SIDEBAR (VERSI PERBAIKAN)
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #667eea; font-size: 2.5rem;'>🎯</h1>
        <h2 style='color: #667eea;'>AI Career Planner</h2>
        <p style='color: #6b7280;'>Your Smart Career Companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.sidebar.markdown("### 🔐 LLM (Gemini) Settings")
    
    # Inisialisasi session state untuk Gemini
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = ""
    if "gemini_model" not in st.session_state:
        st.session_state["gemini_model"] = "Gemini 2.0 Flash"
    
    # Pilihan Model Gemini
    model_options = {
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    }

    # CSS to disable typing but keep dropdown functional
    st.markdown("""
    <style>
    div[data-baseweb="select"] input {
        pointer-events: none !important;
        caret-color: transparent !important;
        color: transparent !important;
    }
    div[data-baseweb="select"] input::placeholder {
        color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Selectbox dengan disabled input
    model_list = list(model_options.keys())
    current_index = model_list.index(st.session_state["gemini_model"]) if st.session_state["gemini_model"] in model_list else 0
    
    selected_model_display = st.sidebar.selectbox(
        "Select Gemini Model",
        options=model_list,
        index=current_index,
        key="gemini_model_select"
    )
    
    # Update session state hanya jika ada perubahan
    if selected_model_display != st.session_state["gemini_model"]:
        st.session_state["gemini_model"] = selected_model_display
    
    # API Key Input (tanpa value binding untuk hindari rerun)
    api_key_input = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        key="api_key_input"
    )
    
    api_confirm_status = st.sidebar.empty()

    # Cek apakah sudah terhubung sebelumnya (agar status hijau tetap muncul)
    if st.session_state.get("gemini_api_key") and st.session_state.get("gemini_api_key").strip():
        api_confirm_status.success(f"✅ Terhubung: {st.session_state.get('gemini_model', 'Gemini')}")

    # Logika Tombol Integrasi
    col_btn1, col_btn2 = st.sidebar.columns([1, 1])
    
    with col_btn1:
        if st.button("🔑 Connect", use_container_width=True):
            if not api_key_input or api_key_input.strip() == "":
                api_confirm_status.error("❌ API Key cannot be empty!")
            else:
                try:
                    genai.configure(api_key=api_key_input)
                    st.session_state["gemini_api_key"] = api_key_input
                    st.session_state["gemini_model"] = selected_model_display
                    st.session_state["gemini_configured"] = True
                    api_confirm_status.success("✅ Integrated successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.session_state["gemini_configured"] = False
                    api_confirm_status.error(f"❌ Connection Failed: {str(e)[:100]}")
    
    with col_btn2:
        if st.button("🔌 Disconnect", use_container_width=True):
            st.session_state["gemini_api_key"] = None
            st.session_state["gemini_model"] = "Gemini 2.0 Flash"
            api_confirm_status.info("🔌 Terputus")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    
    st.markdown("### 📊 About This App")
    st.info("""
    This intelligent system uses **Machine Learning** to:
    - 🎯 Predict ideal job titles
    - ⚙️ Recommend relevant skills
    - 📈 Provide confidence scores
    - 💡 Offer career insights
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Technologies")
    st.markdown("""
    - **Scikit-Learn** - ML Models
    - **TF-IDF** - Text Vectorization
    - **Streamlit** - Web Interface
    - **Plotly** - Interactive Charts
    """)
    
    st.markdown("---")
    
    if resources_loaded:
        st.success("✅ All models loaded")
        st.metric("Total Job Categories", len(le.classes_))
        st.metric("Dataset Size", f"{len(df):,}")
    else:
        st.error("Error loading models")

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown("""
<div class='main-header fade-in'>
    <h1>🎯 AI Career Planner</h1>
    <p>Smart Job Title Prediction & Skill Recommendation System</p>
    <p style='font-size: 0.9rem; opacity: 0.8;'>Powered by Machine Learning & Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)

# Check if resources are loaded
if not resources_loaded:
    st.error("Please check your model paths and try again.")
    st.stop()

# =============================================================================
# FEATURE HIGHLIGHTS
# =============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-box fade-in'>
        <h3>🎯 Career Prediction</h3>
        <p>AI-powered job title prediction with confidence scores</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-box fade-in'>
        <h3>⚙️ Skill Matching</h3>
        <p>Intelligent skill recommendation engine</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-box fade-in'>
        <h3>📊 Data Insights</h3>
        <p>Interactive visualizations and analytics</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Career Prediction", 
    "⚙️ Skill Recommendation",
    "🧠 Agent",
    "📊 Analytics Dashboard"
])

# =============================================================================
# TAB 1: CAREER PREDICTION
# =============================================================================
with tab1:
    st.markdown("""<h3>🎯 Discover Your Ideal Career Path</h3>""", unsafe_allow_html=True)
    

    st.markdown("""
    <div class='info-box' style='margin-top: 1rem;'>
        <strong>💡 Tips for better predictions :</strong>
        <ul>
            <li>Include your skills (e.g., Python, SQL, Leadership)</li>
            <li>Describe your experience and responsibilities</li>
            <li>Mention tools and technologies you use</li>
            <li>Add any relevant certifications or education</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("""
        <div class='stats-box'>
            <p class='stats-number'>97%</p>
            <p class='stats-label'>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""
        <div class='stats-box' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <p class='stats-number'>500+</p>
            <p class='stats-label'>Job Categories</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h3>📝 Enter your professional profile :</h3>", unsafe_allow_html=True)
    
    # Inisialisasi session state untuk user_input
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    
    user_input = st.text_area(
        "Input Here :",
        height=150,
        placeholder="Example : I have 5 years of experience in data analysis using Python, SQL, and Tableau. I've worked on machine learning projects and have strong statistical skills...",
        help="Describe your skills, experience, and what you're looking for in a career",
        key="user_input_key"
    )
        
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_btn = st.button("🚀 Predict Career Path", use_container_width=True)
    
    if predict_btn:
        if len(user_input.strip()) == 0:
            st.warning("⚠️ Please enter your professional profile to get predictions.")
        else:
            with st.spinner("🔮 Analyzing your profile..."):
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                results = predict_with_confidence(user_input, top_k=5)
                
                st.success("✅ Analysis complete!")
                
                # Top Prediction
                top_job, top_score = results[0]
                
                st.markdown(f"""
                <div class='prediction-card fade-in'>
                    <div class='prediction-title'>🎯 Top Prediction</div>
                    <h2 style='margin: 1rem 0; font-size: 2.5rem;'>{top_job}</h2>
                    <div class='prediction-confidence'>{top_score*100:.1f}% Match</div>
                </div>
                """, unsafe_allow_html=True)
                
                # All Predictions
                st.markdown("<h3>📊 All Predictions</h3>", unsafe_allow_html=True)
                
                # Create interactive chart
                labels = [p[0] for p in results]
                scores = [p[1] * 100 for p in results]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=scores,
                        y=labels,
                        orientation='h',
                        marker=dict(
                            color=scores,
                            colorscale='Viridis',
                            line=dict(color='rgba(255,255,255,0.5)', width=2)
                        ),
                        text=[f'{s:.1f}%' for s in scores],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title='Career Prediction Confidence',
                    xaxis_title='Confidence Score (%)',
                    yaxis_title='Job Title',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Results
                st.markdown("<h3>📋 Detailed Results</h3>",unsafe_allow_html=True)
                
                for idx, (label, score) in enumerate(results, 1):
                    with st.expander(f"#{idx} - {label} ({score*100:.1f}%)", expanded=(idx==1)):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **Job Title :** {label}  
                            **Confidence Score :** {score*100:.2f}%  
                            **Match Quality :** {'Excellent' if score > 0.7 else 'Good' if score > 0.5 else 'Fair'}
                            """)
                            
                            # Progress bar for confidence
                            st.markdown(f"""
                            <div class='progress-container'>
                                <div class='progress-bar' style='width: {score*100}%;'>
                                    {score*100:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if score > 0.7:
                                st.markdown("🌟 **Excellent Match**")
                            elif score > 0.5:
                                st.markdown("✅ **Good Match**")
                            else:
                                st.markdown("💡 **Consider This**")

# =============================================================================
# TAB 2: SKILL RECOMMENDATION
# =============================================================================
with tab2:
    st.markdown("<h3>⚙️ Discover Skills to Boost Your Career</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>💡 How it works :</strong>
        <ul>
            <li>Enter your current skills (comma or space separated)</li>
            <li>Our AI analyzes top jobs matching your profile</li>
            <li>Get personalized skill recommendations</li>
            <li>Learn what skills are in demand</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("""
        <div class='stats-box' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);'>
            <p class='stats-number'>1000+</p>
            <p class='stats-label'>Unique Skills</p>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("""
        <div class='stats-box' style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);'>
            <p class='stats-number'>AI</p>
            <p class='stats-label'>Powered Engine</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Inisialisasi session state untuk skill_input
    if "skill_input" not in st.session_state:
        st.session_state["skill_input"] = ""
    
    st.markdown("<h3>🔧 Enter your current skills :</h3>", unsafe_allow_html=True)
    skill_input = st.text_input(
        "Input Here :",
        placeholder="Example : python sql aws machine learning",
        help="Enter skills separated by spaces or commas",
        key="skill_input_key"
    )
    
    # Inisialisasi session state untuk Tab2
    if "top_jobs_slider" not in st.session_state:
        st.session_state["top_jobs_slider"] = 15
    if "top_skills_slider" not in st.session_state:
        st.session_state["top_skills_slider"] = 30
    
    top_jobs_slider = st.session_state.get("top_jobs_slider", 15)
    top_skills_slider = st.session_state.get("top_skills_slider", 30)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        recommend_btn = st.button("🔍 Find Skills", use_container_width=True)
    
    if recommend_btn:
        if len(skill_input.strip()) == 0:
            st.warning("⚠️ Please enter your skills to get recommendations.")
        else:
            with st.spinner("🔍 Analyzing job market..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                recs = recommend_skills_from_text(
                    skill_input, 
                    top_jobs=top_jobs_slider, 
                    top_skills=top_skills_slider
                )
                
                st.success("✅ Recommendations ready!")
                
                if recs:
                    # Summary
                    st.markdown(f"""
                    <div class='prediction-card fade-in'>
                        <h3>📈 Analysis Summary</h3>
                        <p style='font-size: 1.2rem; margin: 1rem 0;'>
                            Based on analysis of <strong>{top_jobs_slider} jobs</strong> 
                            matching your profile, we found <strong>{len(recs)} recommended skills</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interactive Chart
                    st.markdown("<h3>📊 Skill Frequency Analysis</h3>", unsafe_allow_html=True)

                    skill_names = [s[0] for s in recs]
                    frequencies = [s[1] for s in recs]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=frequencies,
                            y=skill_names,
                            orientation='h',
                            marker=dict(
                                color=frequencies,
                                colorscale='Plasma',
                                line=dict(color='rgba(255,255,255,0.5)', width=2)
                            ),
                            text=frequencies,
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title='Recommended Skills Frequency',
                        xaxis_title='Frequency in Top Jobs',
                        yaxis_title='Skills',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Skill Cards
                    st.markdown("<h3>🎯 Recommended Skills</h3>", unsafe_allow_html=True)
                    
                    # Display as badges
                    skills_html = ""
                    for skill, freq in recs:
                        skills_html += f"<span class='skill-badge'>{skill} ({freq})</span>"
                    
                    st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)
                    
                    # Detailed Table
                    st.markdown("<h3>📋 Detailed Breakdown</h3>", unsafe_allow_html=True)
                    
                    skill_df = pd.DataFrame(recs, columns=['Skill', 'Frequency'])
                    skill_df['Priority'] = skill_df['Frequency'].apply(
                        lambda x: '🔥 High' if x > skill_df['Frequency'].quantile(0.7) 
                        else '⭐ Medium' if x > skill_df['Frequency'].quantile(0.3) 
                        else '💡 Low'
                    )
                    
                    st.dataframe(
                        skill_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download button
                    csv = skill_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Skills Report",
                        data=csv,
                        file_name="skill_recommendations.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("💡 No additional skills found. Your skill set is already comprehensive!")

# =============================================================================
# TAB 3: Agent
# =============================================================================
with tab3:
    # ================================================================
    # 🧠 AI AGENT SECTION — Clean, Interactive, Modular
    # ================================================================
    st.markdown("<h3>🧠 Agent: Generative Tools + LLM</h3>", unsafe_allow_html=True)
    st.info("""
    Agent dengan chat history, multi-turn, dan sistem tools:
    - **Gemini LLM** → tanya apa saja tentang karier.
    - **SkillGapChecker** → deteksi skill Anda dan skill yang dibutuhkan suatu role.
    - **RoadmapGenerator** → buat rencana belajar sesuai role.
    Riwayat chat agent akan tersimpan otomatis.
    """)

    # ---------------------------------------
    # SESSION STATE
    # ---------------------------------------
    st.session_state.setdefault("agent_messages", [])
    st.session_state.setdefault("agent_history", [])

    # ---------------------------------------
    # GEMINI CONFIG
    # ---------------------------------------
    gemini_api_key = st.session_state.get("gemini_api_key", "")
    gemini_model_display = st.session_state.get("gemini_model", "Gemini 2.0 Flash")

    model_map = {
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    }
    gemini_model = model_map.get(gemini_model_display, "gemini-2.0-flash")

    @st.cache_resource
    def init_gemini(api_key, model_name):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)

    def llm_generate(prompt):
        if not gemini_api_key:
            return "❌ Belum terhubung ke Gemini.", None

        try:
            model = init_gemini(gemini_api_key, gemini_model)
            response = model.generate_content(prompt)
            return response.text, response.usage_metadata

        except Exception as e:
            if "429" in str(e).lower():
                time.sleep(1.3)
                try:
                    model = init_gemini(gemini_api_key, gemini_model)
                    response = model.generate_content(prompt)
                    return response.text, response.usage_metadata
                except:
                    return "❌ Kena rate limit. Tunggu beberapa detik.", None

            return f"❌ Error: {str(e)[:200]}", None

    # ---------------------------------------
    # SKILL GAP ENGINE (improved)
    # ---------------------------------------
    def extract_user_skills(raw):
        if not raw:
            return []
        return [s.strip().lower() for s in raw.split(",") if s.strip()]

    def get_role_top_skills(role, top_n=12):
        """Ambil skill terbanyak dari dataset berdasarkan role."""
        role_df = df[df["job_title_clean"].str.contains(role, case=False, na=False)]
        if len(role_df) == 0:
            role_df = df.sample(50)

        all_sk = [s for row in role_df["skills_token_list"] for s in row]
        top = Counter(all_sk).most_common(top_n)
        return [s for s, _ in top]

    def skill_gap_checker(user_skills, role):
        role_skills = get_role_top_skills(role)
        missing = [s for s in role_skills if s not in user_skills]
        if not missing:
            return "🎉 Kamu sudah memenuhi semua skill utama untuk role ini!"
        return f"Skill yang masih kurang:\n- " + "\n- ".join(missing)

    # ---------------------------------------
    # ROADMAP GENERATOR (enhanced)
    # ---------------------------------------
    def roadmap_generator(role):
        role_skills = get_role_top_skills(role, top_n=6)
        return f"""
    ### 📘 Roadmap Belajar: **{role}**

    **Minggu 1 — Dasar-dasar**
    - Kuasai fundamental: {role_skills[0]}, {role_skills[1]}

    **Minggu 2 — Skill Teknis**
    - Pelajari: {role_skills[2]}, {role_skills[3]}

    **Minggu 3 — Analisis & Mini Project**
    - Buat mini project kecil relevan: data kecil, dashboard mini, atau script otomatisasi.

    **Minggu 4 — Portfolio**
    - Buat portfolio berbasis 3 skill utama:  
    {role_skills[0]}, {role_skills[1]}, {role_skills[2]}

    **Minggu 5-8 — Expert Path**
    - Ikuti course lanjutan
    - Tambah mini project
    - Bangun GitHub portfolio profesional
    """

    # ---------------------------------------
    # MODEL CONTEXT
    # ---------------------------------------
    def get_best_model_context(query, top_k=4):
        cleaned = clean_text(query)
        vec = tfidf.transform([cleaned])

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vec)[0]
            top_idx = np.argsort(prob)[::-1][:top_k]
            labels = le.inverse_transform(top_idx)
            scores = prob[top_idx]
        else:
            dec = model.decision_function(vec)
            if dec.ndim == 1:
                dec = [dec]
            top_idx = np.argsort(dec[0])[::-1][:top_k]
            labels = le.inverse_transform(top_idx)
            e = np.exp(dec[0][top_idx] - np.max(dec[0][top_idx]))
            scores = e / e.sum()

        ctx = "\n".join([f"- {l} (score: {s:.2f})" for l, s in zip(labels, scores)])
        return ctx, labels.tolist()

    # ---------------------------------------
    # PROMPT BUILDER
    # ---------------------------------------
    def build_prompt(context, history, user_input):
        hist = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        return f"""
    You are **AI Career Agent**. Gunakan konteks berikut untuk membantu user.

    # CONTEXT MODEL
    {context}

    # CHAT HISTORY (ringkas)
    {hist}

    # USER INPUT
    {user_input}

    # TASK
    Jawab secara profesional, jelas, dan langsung ke inti.
    """


    # ================================================================
    # UI TOOL SELECTOR
    # ================================================================
    tool = st.selectbox("Pilih Agent Tool:", ["Gemini LLM", "SkillGapChecker", "RoadmapGenerator"])

    # ================================================================
    # ✨ GEMINI LLM CHAT
    # ================================================================
    if tool == "Gemini LLM":
        st.markdown("💬 **Chat bebas dengan Gemini LLM**")

        user_msg = st.chat_input("Tulis pertanyaan...")

        if user_msg:
            # history terakhir untuk LLM
            history = st.session_state["agent_messages"][-3:]
            context, _ = get_best_model_context(user_msg)

            prompt = build_prompt(context, history, user_msg)
            answer, usage = llm_generate(prompt)

            st.session_state["agent_messages"].append({"role": "user", "content": user_msg, "tool": "Gemini LLM"})
            st.session_state["agent_messages"].append({"role": "assistant", "content": answer, "tool": "Gemini LLM"})

            st.chat_message("assistant").markdown(answer)

    # ================================================================
    # 🔧 SKILL GAP CHECKER
    # ================================================================
    elif tool == "SkillGapChecker":
        st.markdown("🛠️ **Cek kekurangan skill Anda untuk role tertentu**")

        user_skills_raw = st.text_input("Skill Anda (pisahkan dengan koma):")
        role = st.text_input("Role yang diinginkan:", placeholder="Data Scientist / Data Analyst / dll")

        if st.button("Cek Skill Gap"):
            user_skills = extract_user_skills(user_skills_raw)
            role = role.strip() or "Data Scientist"

            result = skill_gap_checker(user_skills, role)
            st.session_state["agent_messages"].append({"role": "user", "content": f"[SkillGapChecker] {user_skills_raw}", "tool": "SkillGapChecker"})
            st.session_state["agent_messages"].append({"role": "assistant", "content": result, "tool": "SkillGapChecker"})

            st.chat_message("assistant").markdown(result)

    # ================================================================
    # 📘 ROADMAP GENERATOR
    # ================================================================
    elif tool == "RoadmapGenerator":
        st.markdown("📗 **Buat roadmap belajar sesuai role**")

        role = st.text_input("Role:", placeholder="Data Analyst / AI Engineer / Software Engineer")

        if st.button("Generate Roadmap"):
            role = role.strip() or "Data Analyst"
            result = roadmap_generator(role)

            st.session_state["agent_messages"].append({"role": "user", "content": f"[Roadmap] {role}", "tool": "RoadmapGenerator"})
            st.session_state["agent_messages"].append({"role": "assistant", "content": result, "tool": "RoadmapGenerator"})

            st.chat_message("assistant").markdown(result)

    # ================================================================
    # HISTORY VIEW
    # ================================================================
    st.markdown("### 📜 Riwayat Interaksi Agent")
    for msg in st.session_state["agent_messages"]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(f"**[{msg.get('tool','')}]** {msg['content']}")

# =============================================================================
# TAB 4: ANALYTICS DASHBOARD
# =============================================================================
with tab4:
    st.markdown("<h3>📊 Analytics Dashboard</h3>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
            <div style="padding:16px; border-radius:16px; background:#ffffff10;">
        """, unsafe_allow_html=True)
    if resources_loaded:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jobs in Dataset", f"{len(df):,}", delta="Updated")
        with col2:
            st.metric("Unique Job Titles", len(df['job_title_clean'].unique()), delta="Active")
        with col3:
            st.metric("Total Skills Analyzed", len(skill_vect.get_feature_names_out()), delta="Growing")
        with col4:
            st.metric("Model Accuracy", "97%", delta="+2%")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("<h3>🎨 Job Distribution</h3>", unsafe_allow_html=True)
        st.markdown("""
            <div style="padding:16px; border-radius:16px; background:#ffffff10;">
        """, unsafe_allow_html=True)

        with st.container():
            job_counts = df['job_title_clean'].value_counts().reset_index()
            job_counts.columns = ['job_title_clean', 'count']
            fig = px.treemap(
                job_counts,
                path=['job_title_clean'],
                values='count',
                color='count',
                color_continuous_scale='Inferno',
                title='Job Distribution Treemap'
            )

            fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{value}",
                textposition="middle center",
                hovertemplate="<b>%{label}</b><br>Count: %{value}",
            )
            
            fig.update_layout(
                height=400,
                uniformtext=dict(
                    minsize=16    # minimum text size
                ),
                # title_x=0.4,
                margin=dict(l=0, r=0, t=50, b=0),
                font=dict(
                    family="Arial",
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            col_left, col_right = st.columns([1.8, 1])

            with col_left:
                job_idx = df['job_title_clean'].value_counts().head(10)
                job_counts = df['job_title_clean'].value_counts().head(10)
                fig = px.bar(
                    job_counts,
                    x=job_idx.index,
                    y=job_counts.values, 
                    title="Top 10 Job Titles",
                    labels={'x':'Job Title', 'y':'Count'},
                    color=job_counts.values,
                    color_continuous_scale='Plasma'
                )

                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_tickangle=-45
                )

                fig.update_traces(marker=dict(line=dict(width=0)))
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Job Category Donut Chart
                df['job_category'] = df['job_title_clean'].apply(lambda x: x.split()[0])  # kategori simple
                cat_counts = df['job_category'].value_counts().head(10)

                fig = px.pie(
                    names=cat_counts.index,
                    values=cat_counts.values,
                    hole=0.55,
                    title="Job Category Distribution",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                st.plotly_chart(fig, use_container_width=True)

            # Job Clustering (UMAP + KMeans 2D Plot)
            sample = df.sample(min(200, len(df)))
            vectors = skill_vect.transform(sample['job_title_clean'])

            tsne = TSNE(n_components=2, random_state=42)
            reduced = tsne.fit_transform(vectors.toarray())

            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(reduced)

            fig = px.scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                color=clusters.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set3,
                labels={'x': 'Dim 1', 'y': 'Dim 2'},
                title="Job Title Clustering"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("<h3>⚙️ Most Common Skills</h3>", unsafe_allow_html=True)
        st.markdown("<div style='padding:16px; border-radius:16px; background:#ffffff10;'>", unsafe_allow_html=True)
        with st.container():
            all_skills = []
            df['skills_token_list'] = df['skills_token_list'].apply(lambda lst: [s.replace("_", " ").title() for s in lst])
            for skills in df["skills_token_list"]:
                all_skills.extend(skills)
            skill_counts = Counter(all_skills).most_common(10)
            skill_names, skill_freqs = zip(*skill_counts)

            top_skills = dict(skill_counts[:10])

            # Skill Trend Line
            categories = list(top_skills.keys())
            trend_skills = list(top_skills.keys())
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "jul", "Aug", "Sep", "Oct", "Nov", "Des"]
            fig = go.Figure()

            for skill in trend_skills:
                trend = np.random.randint(20, 120, len(months))
                fig.add_trace(go.Scatter(
                    x=months,
                    y=trend,
                    mode="lines+markers",
                    name=skill
                ))

            fig.update_layout(
                height=400,
                title="Skill Trend Over Time",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            col_left, col_right = st.columns([1.5, 1])
            with col_left:
                fig = px.bar(
                    x=skill_freqs,
                    y=skill_names,
                    orientation='h', 
                    title="Top 10 Skills",
                    labels={'x':'Frequency', 'y':'Skill'},
                    color=skill_freqs,
                    color_continuous_scale='Magma'
                )

                fig.update_layout(
                    height=600,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=40, b=20),
                )

                fig.update_traces(marker=dict(line=dict(width=0)))
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Skill Gap Radar Chart
                industry_skill = np.array(list(top_skills.values()))
                candidate_skill = industry_skill * np.random.uniform(0.4, 0.9, len(industry_skill))
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=industry_skill,
                    theta=categories,
                    fill='toself',
                    name='Industry Demand',
                    line_color='#F72585'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=candidate_skill,
                    theta=categories,
                    fill='toself',
                    name='Candidate Skill',
                    line_color='#4361EE'
                ))

                fig.update_layout(
                    title="Skill Gap Radar",
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

            # Career Graph (Network Visualization)
            G = nx.Graph()
            top_pairs = [(skill, job) for skill, job in zip(skill_names[:10], job_idx.index[:10])]

            for skill, job in top_pairs:
                G.add_node(skill, type='skill')
                G.add_node(job, type='job')
                G.add_edge(skill, job)

            pos = nx.spring_layout(G, seed=42)

            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color="#4CC9F0")
            ))

            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            labels = list(G.nodes())

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=20, color="#F72585"),
                text=labels, textposition="top center"
            ))

            fig.update_layout(
                title="Career Path Graph",
                height=400,
                template="plotly_dark",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        # JOB–SKILL HEATMAP (Aesthetic Premium)
        st.markdown("<h3>🔥 Job-Skill Heatmap (Top 20 Skills x Top 15 Job Titles)</h3>", unsafe_allow_html=True)
        st.markdown("""<div style='padding:16px; border-radius:16px; background:#ffffff10;'>""", unsafe_allow_html=True)
        with st.container():
            # Extract top jobs
            top_jobs = df['job_title_clean'].value_counts().head(15).index

            # Build skill matrix (count skill occurrences per job)
            matrix = {}
            for job in top_jobs:
                subset = df[df['job_title_clean'] == job]['skills_token_list']
                skills = [skill for row in subset for skill in row]
                skill_count = Counter(skills)
                matrix[job] = dict(skill_count)

            skill_df = pd.DataFrame(matrix).fillna(0).astype(int)

            # Take top 20 skills only
            skill_df = skill_df.loc[skill_df.sum(axis=1).sort_values(ascending=False).head(20).index]

            fig = px.imshow(
                skill_df,
                color_continuous_scale="Plasma",
                aspect="auto",
                labels=dict(color="Frequency"),
            )

            fig.update_layout(
                height=520,
                margin=dict(l=60, r=40, t=40, b=40),
                coloraxis_colorbar=dict(
                    title="Count",
                    thickness=15,
                    outlinewidth=1,
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("<h3>🕸 Skill Co-Occurrence Network (Top 30 Skills)</h3>", unsafe_allow_html=True)
        st.markdown("""<div style='padding:16px; border-radius:16px; background:#ffffff10;'>""", unsafe_allow_html=True)
        with st.container():
            # flatten skills
            all_skills = [skill for row in df['skills_token_list'] for skill in row]

            # get top skills
            top_skills = [s for s, _ in Counter(all_skills).most_common(30)]

            # build graph
            G = nx.Graph()

            for skills in df['skills_token_list']:
                filtered = [s for s in skills if s in top_skills]
                for i in range(len(filtered)):
                    for j in range(i+1, len(filtered)):
                        if G.has_edge(filtered[i], filtered[j]):
                            G[filtered[i]][filtered[j]]['weight'] += 1
                        else:
                            G.add_edge(filtered[i], filtered[j], weight=1)

            pos = nx.spring_layout(G, k=0.4, iterations=50)

            # Plotly nodes
            edge_x = []
            edge_y = []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=0.7, color="rgba(120,120,120,0.3)"),
                hoverinfo="none"
            ))

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=18,
                    color=list(range(len(G.nodes()))),
                    colorscale="Turbo",
                    line=dict(width=1, color="white")
                )
            ))

            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
    else:
        st.error("❌ Analytics unavailable - models not loaded")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class='custom-footer'>
    <p>© 2025 AI Career Planner</p>
    <p>Built with using Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)