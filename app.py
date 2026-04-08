import streamlit as st
import pickle
import pypdf
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    clean_resume, lemmatize_text, extract_skills,
    DOMAIN_BADGE_CLASS,
)

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
    /* ── Base ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* ── Hero ── */
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
        background-size: 200% auto;
        animation: gradientShift 4s linear infinite;
        margin-bottom: 10px;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero p {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    /* typewriter effect on hero subtitle */
    .hero .typewriter {
        display: inline-block;
        overflow: hidden;
        border-right: 3px solid #a78bfa;
        white-space: nowrap;
        animation: typing 3.5s steps(60,end) forwards,
                   blink 0.75s step-end infinite;
    }
    @keyframes typing {
        from { width: 0; }
        to   { width: 100%; }
    }
    @keyframes blink {
        from, to { border-color: transparent; }
        50%      { border-color: #a78bfa; }
    }

    /* ── Nav bar ── */
    .nav-bar {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin: 0 0 20px 0;
        flex-wrap: wrap;
    }
    .nav-btn {
        padding: 10px 24px;
        border-radius: 50px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.06);
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 600;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.2s;
    }
    .nav-btn:hover {
        background: rgba(99,102,241,0.25);
        border-color: #6366f1;
        color: white;
    }
    .nav-btn.active {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border-color: transparent;
        color: white;
        box-shadow: 0 0 16px rgba(99,102,241,0.5);
    }

    /* ── Metric card ── */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.3);
        border-color: rgba(99,102,241,0.5);
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

    /* ── Result box ── */
    .result-box {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(99,102,241,0.4);
        animation: fadeInUp 0.4s ease;
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
    .result-box .conf-pill {
        display: inline-block;
        margin-top: 12px;
        padding: 4px 16px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        background: rgba(255,255,255,0.2);
        color: white;
    }

    /* ── Step card ── */
    .step-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #6366f1;
        margin-bottom: 10px;
        transition: border-color 0.2s;
    }
    .step-card:hover {
        border-left-color: #34d399;
    }

    /* ── Skill badges ── */
    .badge-languages   { background: rgba(99,102,241,0.3); color: #c4b5fd; }
    .badge-ml          { background: rgba(52,211,153,0.25); color: #6ee7b7; }
    .badge-data        { background: rgba(96,165,250,0.25); color: #93c5fd; }
    .badge-cloud       { background: rgba(251,191,36,0.2);  color: #fcd34d; }
    .badge-web         { background: rgba(244,63,94,0.2);   color: #fda4af; }
    .badge-base {
        display: inline-block;
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px 3px;
        font-size: 0.82rem;
        font-weight: 600;
        vertical-align: middle;
    }

    /* ── Match score colours ── */
    .score-high   { color: #34d399; font-weight: 700; }
    .score-medium { color: #fbbf24; font-weight: 700; }
    .score-low    { color: #f87171; font-weight: 700; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Misc overrides ── */
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
        transition: box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 18px rgba(99,102,241,0.5) !important;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #34d399) !important;
    }
    .stRadio label { color: white !important; }
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
    div[role="radiogroup"] label p { color: white !important; margin: 0 !important; }

    /* ── Toggle label & caption visibility on dark background ── */
    .stToggle label, .stToggle p { color: white !important; }
    [data-testid="stCaptionContainer"] p { color: #c4b5fd !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
@st.cache_resource
def _load_model():
    with open('models/resume_model.pkl', 'rb') as f:
        return pickle.load(f)

model_data = _load_model()

model      = model_data['model']
tfidf      = model_data['tfidf']
le         = model_data['le']
model_name = model_data['model_name']
accuracy   = model_data['accuracy']
cv         = model_data['cv']
categories = model_data['categories']

# ── Helper Functions ──
def extract_text_from_pdf(pdf_file):
    try:
        reader = pypdf.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
    except (pypdf.errors.PdfReadError, pypdf.errors.PdfStreamError, ValueError) as e:
        st.error(f"⚠️ Could not read PDF: {e}")
        return ''

def predict_category(text):
    cleaned   = clean_resume(text)
    processed = lemmatize_text(cleaned)
    # Support both Pipeline format (new) and standalone classifier format (old pkl)
    if hasattr(model, 'named_steps'):
        prediction = model.predict([processed])[0]
        probs      = model.predict_proba([processed])[0]
    else:
        vectorized = tfidf.transform([processed])
        prediction = model.predict(vectorized)[0]
        probs      = model.predict_proba(vectorized)[0]
    category = le.inverse_transform([prediction])[0]
    return category, probs

# ── BERT Model (lazy-loaded, optional) ──
@st.cache_resource(show_spinner=False)
def _load_bert():
    """Load BERT pkl + SentenceTransformer encoder. Returns (bert_data, encoder) or (None, None)."""
    try:
        from sentence_transformers import SentenceTransformer
        with open('models/bert_resume_model.pkl', 'rb') as f:
            bert_data = pickle.load(f)
        encoder = SentenceTransformer(bert_data['bert_model'])
        return bert_data, encoder
    except Exception:
        return None, None

def predict_category_bert(text, bert_data, encoder):
    cleaned   = clean_resume(text)
    processed = lemmatize_text(cleaned)
    embedding = encoder.encode([processed])
    prediction = bert_data['classifier'].predict(embedding)[0]
    probs      = bert_data['classifier'].predict_proba(embedding)[0]
    category   = bert_data['le'].inverse_transform([prediction])[0]
    return category, probs

# ── JD matching ──
MAX_RESUMES = 5

def compute_jd_match(jd_text, resume_text):
    jd_vec  = tfidf.transform([lemmatize_text(clean_resume(jd_text))])
    res_vec = tfidf.transform([lemmatize_text(clean_resume(resume_text))])
    score   = cosine_similarity(jd_vec, res_vec)[0][0]
    return float(score) * 100

# ── Hero Section ──
st.markdown("""
<div class="hero">
    <h1>🤖 Resume Screening AI</h1>
    <p><span class="typewriter">Powered by NLP &amp; Machine Learning — Instantly predict job categories from any resume</span></p>
</div>
""", unsafe_allow_html=True)

# ── Navigation ──
if 'page' not in st.session_state:
    st.session_state.page = "screen"

_pages = [
    ("screen", "📄 Screen Resume"),
    ("match",  "🎯 JD Match"),
    ("stats",  "📊 Model Stats"),
    ("about",  "ℹ️ About"),
]

nav_html = '<div class="nav-bar">'
for key, label in _pages:
    cls = "nav-btn active" if st.session_state.page == key else "nav-btn"
    nav_html += f'<span class="{cls}">{label}</span>'
nav_html += '</div>'
st.markdown(nav_html, unsafe_allow_html=True)

# Actual clickable buttons (hidden behind the styled nav; kept for interactivity)
_nav_cols = st.columns(len(_pages))
for i, (key, label) in enumerate(_pages):
    with _nav_cols[i]:
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()

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

        st.markdown("**🧠 Model**")
        use_bert = st.toggle("Use BERT (all-MiniLM-L6-v2)", value=False,
                             help="TF-IDF is fast and deterministic. "
                                  "BERT uses semantic sentence embeddings (384-dim) "
                                  "for richer generalisation on unseen resume styles.")
        if use_bert:
            st.caption("⚡ BERT model loads on first use (~90 MB download)")

        predict_btn = st.button("🔍 Analyze Resume")

    with col_right:
        st.markdown("### 🎯 Results")

        if predict_btn:
            if not resume_text.strip():
                st.error("⚠️ Please upload a PDF or paste resume text first!")
            else:
                if use_bert:
                    with st.spinner("🧠 Loading BERT model & generating embeddings…"):
                        bert_data, bert_encoder = _load_bert()
                    if bert_data is None:
                        st.error("⚠️ BERT model unavailable — `sentence-transformers` is not installed. "
                                 "Falling back to TF-IDF.")
                        category, probs = predict_category(resume_text)
                        model_label      = model_name + " (TF-IDF fallback)"
                        active_categories = categories
                    else:
                        with st.spinner("🧠 BERT is analyzing your resume…"):
                            category, probs = predict_category_bert(resume_text, bert_data, bert_encoder)
                        model_label       = bert_data['model_name']
                        active_categories = bert_data['categories']
                else:
                    with st.spinner("🧠 AI is analyzing your resume..."):
                        category, probs = predict_category(resume_text)
                    model_label       = model_name
                    active_categories = categories

                top_conf = probs.max() * 100
                conf_label = (
                    "🟢 High confidence" if top_conf >= 70
                    else "🟡 Medium confidence" if top_conf >= 40
                    else "🔴 Low confidence"
                )
                st.markdown(f"""
                <div class="result-box">
                    <h2>Predicted Job Category</h2>
                    <h1>{category}</h1>
                    <span class="conf-pill">{conf_label} &mdash; {top_conf:.1f}%</span><br>
                    <span style='font-size:0.8rem;opacity:0.7;margin-top:6px;display:inline-block'>
                        via {model_label}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Top 5 Matching Categories")
                top5_idx = probs.argsort()[-5:][::-1]
                for i, idx in enumerate(top5_idx):
                    cat  = active_categories[idx]
                    prob = probs[idx] * 100
                    emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                    st.markdown(f"{emoji} **{cat}** — `{prob:.1f}%`")
                    st.progress(int(prob))

                skills = extract_skills(resume_text)
                if skills:
                    st.markdown("#### 🛠️ Skills Detected")
                    for domain, skill_list in skills.items():
                        cls = DOMAIN_BADGE_CLASS.get(domain, "badge-languages")
                        badges = " ".join(
                            f"<span class='badge-base {cls}'>{s}</span>"
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
        f"<p style='color:#94a3b8'>Paste a Job Description and upload up to {MAX_RESUMES} resumes "
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
        st.markdown(f"#### 📤 Upload Resumes (up to {MAX_RESUMES} PDFs)")
        uploaded_resumes = st.file_uploader(
            "", type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="resume_uploads"
        )
        if uploaded_resumes:
            count = min(len(uploaded_resumes), MAX_RESUMES)
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
                for f in uploaded_resumes[:MAX_RESUMES]:
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
                score_cls = (
                    "score-high"   if r["score"] >= 70
                    else "score-medium" if r["score"] >= 40
                    else "score-low"
                )
                with st.expander(
                    f"{medals[i]}  {r['name']}  —  Match: {r['score']:.1f}%",
                    expanded=(i == 0)
                ):
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.progress(int(min(r["score"], 100)))
                        st.markdown(
                            f"**Match score:** <span class='{score_cls}'>{r['score']:.1f}%</span>"
                            f"&nbsp;&nbsp;|&nbsp;&nbsp;**Predicted Category:** `{r['category']}`",
                            unsafe_allow_html=True
                        )
                    with c2:
                        if r["skills"]:
                            st.markdown("**Skills found:**")
                            badges_html = ""
                            for domain, slist in r["skills"].items():
                                cls = DOMAIN_BADGE_CLASS.get(domain, "badge-languages")
                                for s in slist[:4]:
                                    badges_html += f"<span class='badge-base {cls}'>{s}</span> "
                            st.markdown(badges_html, unsafe_allow_html=True)

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
        (384 dimensions vs 1500 TF-IDF features) and would generalize
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
            ("1", "Text Extraction",     "Reads PDF or accepts pasted text"),
            ("2", "Text Cleaning",       "Removes URLs, symbols, special characters"),
            ("3", "Stopword Filtering",  "Removes common words using sklearn ENGLISH_STOP_WORDS"),
            ("4", "Vectorization",       "TF-IDF (1500 features) or BERT semantic embeddings (384-dim) — selectable via toggle"),
            ("5", "ML Classification",   "Best Pipeline predicts from 25 categories"),
            ("6", "Skill Detection",     "Regex-based matching with abbreviation support across 5 domains"),
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
            ("🐍", "Python 3.11",             "Core language"),
            ("🔤", "sklearn",                  "Stopword filtering (ENGLISH_STOP_WORDS)"),
            ("🔢", "TF-IDF",                   "Text vectorization (1500 features)"),
            ("🤗", "sentence-transformers",    "BERT embeddings via all-MiniLM-L6-v2 (384-dim)"),
            ("🌲", "Scikit-learn",             "ML pipelines, evaluation & cosine similarity"),
            ("📄", "pypdf",                    "PDF text extraction"),
            ("🚀", "Streamlit",                "Web app framework"),
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