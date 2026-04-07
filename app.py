import streamlit as st
import pickle
import re
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Page Config ──
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .hero {
        text-align: center;
        padding: 40px 20px 20px 20px;
    }
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero p {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-card h2 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .result-box {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .result-box h2 {
        color: white;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
        opacity: 0.8;
    }
    .result-box h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .step-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #6366f1;
        margin-bottom: 10px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 30px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #34d399) !important;
    }
    .stRadio label {
        color: white !important;
    }
    .stRadio div {
        background: transparent !important;
        display: flex !important;
        flex-direction: row !important;
        gap: 15px !important;
    }
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 15px !important;
    }
    div[role="radiogroup"] label {
        color: white !important;
        font-size: 0.95rem !important;
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 8px 20px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        cursor: pointer !important;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(99,102,241,0.3) !important;
        border-color: #6366f1 !important;
    }
    div[role="radiogroup"] label p {
        color: white !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
with open('models/resume_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model      = model_data['model']
tfidf      = model_data['tfidf']
le         = model_data['le']
model_name = model_data['model_name']
accuracy   = model_data['accuracy']
cv         = model_data['cv']
categories = model_data['categories']

# ── Helper Functions ──
def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def lemmatize_text(text):
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def predict_category(text):
    cleaned    = clean_resume(text)
    processed  = lemmatize_text(cleaned)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)[0]
    probs      = model.predict_proba(vectorized)[0]
    category   = le.inverse_transform([prediction])[0]
    return category, probs

# ── Skill Vocabulary ──
SKILLS_VOCABULARY = {
    "Languages":    ["python", "java", "javascript", "typescript", "c++", "c#",
                     "scala", "kotlin", "swift", "go", "rust", "php", "ruby",
                     "matlab", "julia"],
    "ML / AI":      ["machine learning", "deep learning", "nlp",
                     "computer vision", "tensorflow", "pytorch", "keras",
                     "scikit-learn", "bert", "transformers", "xgboost",
                     "lightgbm", "reinforcement learning", "llm"],
    "Data":         ["sql", "pandas", "numpy", "spark", "hadoop", "kafka",
                     "tableau", "power bi", "postgresql", "mysql", "mongodb",
                     "airflow", "dbt"],
    "Cloud / DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "git",
                       "linux", "terraform", "jenkins"],
    "Web / APIs":   ["react", "django", "flask", "fastapi", "nodejs",
                     "rest api", "graphql", "html", "css"],
}

def extract_skills(text):
    text_lower = text.lower()
    found = {}
    for domain, skills in SKILLS_VOCABULARY.items():
        matched = [s for s in skills if s in text_lower]
        if matched:
            found[domain] = matched
    return found

def compute_jd_match(jd_text, resume_text):
    jd_vec  = tfidf.transform([lemmatize_text(clean_resume(jd_text))])
    res_vec = tfidf.transform([lemmatize_text(clean_resume(resume_text))])
    score   = cosine_similarity(jd_vec, res_vec)[0][0]
    return float(score) * 100

# ── Hero Section ──
st.markdown("""
<div class="hero">
    <h1>🤖 Resume Screening AI</h1>
    <p>Powered by NLP & Machine Learning — Instantly predict job categories from any resume</p>
</div>
""", unsafe_allow_html=True)

# ── Navigation ──
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
with col1:
    if st.button("📄 Screen Resume"):
        st.session_state.page = "screen"
with col2:
    if st.button("🎯 JD Match"):
        st.session_state.page = "match"
with col3:
    if st.button("📊 Model Stats"):
        st.session_state.page = "stats"
with col4:
    if st.button("ℹ️ About"):
        st.session_state.page = "about"

if 'page' not in st.session_state:
    st.session_state.page = "screen"

st.markdown("---")

# ══════════════════════════════
# PAGE 1: SCREEN RESUME
# ══════════════════════════════
if st.session_state.page == "screen":

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📥 Upload Resume")
        input_method = st.radio("",
                                ["📎 Upload PDF", "✏️ Paste Text"],
                                horizontal=True,
                                label_visibility="collapsed")
        resume_text = ""

        if input_method == "📎 Upload PDF":
            uploaded_file = st.file_uploader("",
                type=['pdf'], label_visibility="collapsed")
            if uploaded_file:
                resume_text = extract_text_from_pdf(uploaded_file)
                st.success(f"✅ PDF loaded — {len(resume_text)} characters extracted")
        else:
            resume_text = st.text_area("",
                height=250,
                placeholder="Paste your resume content here...",
                label_visibility="collapsed")

        predict_btn = st.button("🔍 Analyze Resume")

    with col_right:
        st.markdown("### 🎯 Results")

        if predict_btn:
            if not resume_text.strip():
                st.error("⚠️ Please upload a PDF or paste resume text first!")
            else:
                with st.spinner("🧠 AI is analyzing your resume..."):
                    category, probs = predict_category(resume_text)

                st.markdown(f"""
                <div class="result-box">
                    <h2>Predicted Job Category</h2>
                    <h1>{category}</h1>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Top 5 Matching Categories")
                top5_idx = probs.argsort()[-5:][::-1]
                for i, idx in enumerate(top5_idx):
                    cat  = categories[idx]
                    prob = probs[idx] * 100
                    emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                    st.markdown(f"{emoji} **{cat}** — `{prob:.1f}%`")
                    st.progress(int(prob))

                skills = extract_skills(resume_text)
                if skills:
                    st.markdown("#### 🛠️ Skills Detected")
                    for domain, skill_list in skills.items():
                        badges = " &nbsp;".join(
                            f"<code style='background:rgba(99,102,241,0.25);"
                            f"border-radius:4px;padding:2px 6px'>{s}</code>"
                            for s in skill_list
                        )
                        st.markdown(
                            f"<span style='color:#94a3b8'><strong>{domain}:</strong></span> {badges}",
                            unsafe_allow_html=True
                        )
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px;
                        color:#64748b; border: 2px dashed #334155;
                        border-radius:16px;'>
                <div style='font-size:3rem'>🤖</div>
                <div style='font-size:1.1rem; margin-top:10px'>
                    Upload a resume to see AI predictions
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 2: JD MATCH RANKER
# ══════════════════════════════
elif st.session_state.page == "match":
    st.markdown("### 🎯 Resume–JD Match Ranker")
    st.markdown(
        "<p style='color:#94a3b8'>Paste a Job Description and upload up to 5 resumes "
        "— they are ranked by cosine similarity so you can see who fits best.</p>",
        unsafe_allow_html=True
    )

    col_jd, col_uploads = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("#### 📋 Job Description")
        jd_text = st.text_area(
            "", height=300,
            placeholder="Paste job description here...",
            label_visibility="collapsed",
            key="jd_input"
        )

    with col_uploads:
        st.markdown("#### 📤 Upload Resumes (up to 5 PDFs)")
        uploaded_resumes = st.file_uploader(
            "", type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="resume_uploads"
        )
        if uploaded_resumes:
            count = min(len(uploaded_resumes), 5)
            st.info(f"✅ {count} resume(s) ready")
        match_btn = st.button("🔍 Rank Resumes")

    if match_btn:
        if not jd_text.strip():
            st.error("⚠️ Please paste a job description first!")
        elif not uploaded_resumes:
            st.error("⚠️ Please upload at least one resume PDF!")
        else:
            st.markdown("---")
            st.markdown("### 📊 Ranking Results")
            with st.spinner("🧠 Computing match scores..."):
                ranked = []
                for f in uploaded_resumes[:5]:
                    resume_text = extract_text_from_pdf(f)
                    score    = compute_jd_match(jd_text, resume_text)
                    category, _ = predict_category(resume_text)
                    skills   = extract_skills(resume_text)
                    ranked.append({
                        "name": f.name,
                        "score": score,
                        "category": category,
                        "skills": skills,
                    })
                ranked.sort(key=lambda x: x["score"], reverse=True)

            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            for i, r in enumerate(ranked):
                with st.expander(
                    f"{medals[i]}  {r['name']}  —  Match: {r['score']:.1f}%",
                    expanded=(i == 0)
                ):
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.progress(int(min(r["score"], 100)))
                        st.markdown(f"**Predicted Category:** `{r['category']}`")
                    with c2:
                        if r["skills"]:
                            all_skills = [s for v in r["skills"].values() for s in v]
                            st.markdown("**Skills found:**")
                            st.markdown(
                                " &nbsp;".join(
                                    f"<code style='background:rgba(99,102,241,0.25);"
                                    f"border-radius:4px;padding:2px 6px'>{s}</code>"
                                    for s in all_skills[:10]
                                ),
                                unsafe_allow_html=True
                            )

            st.markdown("""
            <div style='background:rgba(99,102,241,0.12);
                        border-radius:12px; padding:16px;
                        border:1px solid rgba(99,102,241,0.3);
                        margin-top:20px'>
                <strong>💡 How Match Score Works</strong><br>
                <span style='color:#94a3b8; font-size:0.9rem'>
                TF-IDF cosine similarity measures shared domain vocabulary between the JD
                and each resume. Treat scores as a <em>relative ranking signal</em> — a
                resume with 42% isn't a bad candidate, it just uses fewer of the exact
                keywords in this JD. Combine with the predicted category and skill list
                for a fuller picture.
                </span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 3: MODEL STATS
# ══════════════════════════════
elif st.session_state.page == "stats":
    st.markdown("### 📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card">
        <h3>Best Model</h3><h2>{model_name}</h2></div>""",
        unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <h3>Test Accuracy</h3><h2>{accuracy*100:.1f}%</h2></div>""",
        unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <h3>CV Accuracy</h3><h2>{cv*100:.1f}%</h2></div>""",
        unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card">
        <h3>Categories</h3><h2>{len(categories)}</h2></div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Model Comparison")
        st.image('notebooks/model_comparison.png')
    with col2:
        st.markdown("#### Confusion Matrix")
        st.image('notebooks/confusion_matrix.png')

    st.markdown("#### 🤖 TF-IDF vs BERT Comparison")
    st.image('notebooks/tfidf_vs_bert.png')
    st.markdown("""
    <div style='background:rgba(99,102,241,0.15);
                border-radius:12px; padding:20px;
                border:1px solid rgba(99,102,241,0.3)'>
        <strong>💡 Key Insight</strong><br><br>
        <span style='color:#94a3b8'>
        Both TF-IDF and BERT achieve ~99.6% CV accuracy on this dataset.
        Resume text has highly domain-specific vocabulary making it
        linearly separable — both classical and deep learning approaches
        work exceptionally well. BERT provides richer semantic embeddings
        (768 dimensions vs 1500 TF-IDF features) and would generalize
        better on unseen resume formats.
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### ✅ All 25 Supported Job Categories")
    cols = st.columns(5)
    for i, cat in enumerate(categories):
        cols[i % 5].markdown(f"""
        <div style='background:rgba(99,102,241,0.15);
                    border-radius:8px; padding:6px 10px;
                    margin:3px 0; font-size:0.85rem;
                    border-left:3px solid #6366f1'>
            {cat}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════
elif st.session_state.page == "about":
    st.markdown("### ℹ️ About This Project")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔬 How It Works")
        steps = [
            ("1", "Text Extraction", "Reads PDF or accepts pasted text"),
            ("2", "Text Cleaning", "Removes URLs, symbols, special characters"),
            ("3", "Stopword Removal", "Removes common words like 'the', 'is'"),
            ("4", "Lemmatization", "Reduces words to root form"),
            ("5", "TF-IDF Vectorization", "Converts text to 1500 numerical features"),
            ("6", "ML Classification", "Random Forest predicts from 25 categories"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="step-card">
                <strong>Step {num}: {title}</strong><br>
                <span style='color:#94a3b8; font-size:0.9rem'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📦 Tech Stack")
        techs = [
            ("🐍", "Python 3.11", "Core language"),
            ("📝", "NLTK", "Text preprocessing & lemmatization"),
            ("🔢", "TF-IDF", "Text vectorization (1500 features)"),
            ("🌲", "Scikit-learn", "ML models & evaluation"),
            ("📄", "PyPDF2", "PDF text extraction"),
            ("🚀", "Streamlit", "Web app framework"),
        ]
        for emoji, name, desc in techs:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.05);
                        border-radius:10px; padding:12px;
                        margin-bottom:8px;'>
                <span style='font-size:1.5rem'>{emoji}</span>
                <strong style='margin-left:10px'>{name}</strong><br>
                <span style='color:#94a3b8; font-size:0.85rem;
                             margin-left:10px'>{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.15);
                    border-radius:12px; padding:20px;
                    border:1px solid rgba(99,102,241,0.3);
                    margin-top:15px'>
            <strong>📊 Dataset Stats</strong><br><br>
            <span style='color:#94a3b8'>
            📁 962 real resumes<br>
            🏷️ 25 job categories<br>
            🎯 100% test accuracy<br>
            ✅ 99.6% CV accuracy
            </span>
        </div>""", unsafe_allow_html=True)