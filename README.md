# 🤖 Resume Screening AI

An NLP-powered AI tool that automatically screens resumes and predicts the most suitable job category using Machine Learning and Transformer models (all-MiniLM-L6-v2).

🔗 **Live Demo:** https://aman-resume-screening.streamlit.app
📂 **GitHub:** https://github.com/amanwork5454-debug/RESUME-SCREENING

---

## 📌 Project Highlights
- 📄 Upload PDF resume or paste text directly
- 🎯 Predicts job category from 25 options instantly
- 📊 Shows top 5 matching categories with confidence scores
- 🛠️ **Skill extraction** — auto-detects Languages, ML/AI, Data, Cloud, and Web skills with abbreviation support (e.g. `js` → JavaScript, `k8s` → Kubernetes, `ml` → Machine Learning)
- 🤖 **TF-IDF ↔ BERT toggle** — switch between fast TF-IDF and semantic `all-MiniLM-L6-v2` predictions directly in the UI
- 🔍 **Resume–JD Match Ranker** — paste a Job Description, upload up to 5 resumes, get them ranked by cosine similarity
- 🧪 **22 unit tests** covering text cleaning, stop-word filtering, and skill extraction
- 🚀 Deployed live on Streamlit Cloud

---

## 🔬 How It Works
1. **Text Extraction** — Reads PDF or accepts pasted text
2. **Text Cleaning** — Removes URLs, symbols, special characters
3. **Stopword Filtering** — Removes common words using sklearn ENGLISH_STOP_WORDS
4. **TF-IDF Vectorization** — Converts text to 1500 numerical features
5. **ML Classification** — Predicts job category from 25 options
6. **Skill Detection** — Keyword matching across 5 technology domains
7. **JD Matching** — TF-IDF cosine similarity between job description and resumes

---

## 🧠 ML Model Comparison

### TF-IDF Based Models
| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| **Random Forest** | **100%** | **99.6%** |
| Logistic Regression | 99.5% | 99.3% |
| SVM | 99.5% | 99.6% |

### BERT-Based Models (all-MiniLM-L6-v2)
| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| Logistic Regression + BERT | 99.5% | 99.6% |
| Random Forest + BERT | 99.5% | 99.6% |

> **Key Insight:** Both TF-IDF and BERT achieve ~99.6% CV accuracy.
> `all-MiniLM-L6-v2` produces 384-dimensional semantic sentence embeddings
> and generalizes better on unseen resume formats than 1500 TF-IDF sparse
> features.

---

## ⚠️ Limitations & Honest Analysis

This project demonstrates NLP classification and semantic similarity on a clean,
well-labelled dataset. There are a few things to be aware of:

- **High accuracy is expected on this dataset.** Resume text contains highly
  domain-specific vocabulary (e.g. "VLSI" only appears in Electrical Engineering
  resumes), making the classes nearly linearly separable. Real-world performance
  would be lower on messy, multi-role, or ambiguous resumes.
- **JD match scores are relative, not absolute.** Cosine similarity on TF-IDF
  vectors measures keyword overlap, not semantic understanding. A score of 35% is
  not "bad" — use it to rank candidates relative to each other, not as a pass/fail
  threshold.
- **Skill extraction is keyword-based.** It finds known technology names in plain
  text. It won't catch synonyms (e.g. "Py" for Python) or skills described in
  narrative form.
- **Dataset size.** With 962 resumes across 25 categories (~38 per class), the
  model has not seen the full distribution of real-world resume styles. Collecting
  more diverse data would improve robustness.

---

## 🏷️ 25 Supported Job Categories
Data Science, Java Developer, Python Developer, DevOps Engineer,
Web Designing, HR, Testing, Blockchain, ETL Developer, Hadoop,
Sales, Mechanical Engineer, Database, Electrical Engineering,
Health and fitness, PMO, Business Analyst, DotNet Developer,
Automation Testing, Network Security Engineer, Civil Engineer,
SAP Developer, Advocate, Arts, Operations Manager

---

## 🛠️ Tech Stack
| Technology | Usage |
|------------|-------|
| Python 3.11 | Core language |
| sklearn ENGLISH_STOP_WORDS | Stopword filtering |
| TF-IDF | Text vectorization (1500 features) |
| Scikit-learn | ML pipelines, evaluation, cross validation, cosine similarity |
| HuggingFace | all-MiniLM-L6-v2 transformer model |
| Sentence-Transformers | BERT embeddings (384 dimensions) |
| pypdf | PDF text extraction |
| Streamlit | Web dashboard |
| Streamlit Cloud | Deployment |
| Git, GitHub | Version control |

---

## 📁 Project Structure
```
resume-screening/
├── app.py                          # Streamlit dashboard
├── utils.py                        # Shared NLP utilities (cleaning, skill extraction)
├── requirements.txt                # Pinned Python dependencies
├── README.md                       # Project documentation
├── models/
│   ├── resume_model.pkl            # TF-IDF + Random Forest pipeline
│   └── bert_resume_model.pkl       # all-MiniLM-L6-v2 + best classifier
├── notebooks/
│   ├── 01_preprocessing.py         # Text cleaning & NLP pipeline
│   ├── 02_model.py                 # TF-IDF model training & evaluation
│   ├── 03_bert_model.py            # all-MiniLM-L6-v2 training & comparison
│   ├── category_distribution.png   # Resume count by category
│   ├── confusion_matrix.png        # Model confusion matrix
│   ├── model_comparison.png        # TF-IDF models comparison
│   └── tfidf_vs_bert.png           # TF-IDF vs BERT comparison
├── tests/
│   └── test_utils.py               # 22 unit tests for core utilities
└── data/
    ├── README.md                   # Dataset download instructions
    └── UpdatedResumeDataSet.csv    # Not in GitHub (gitignored)
```

## 📊 Dataset
- **Source:** Kaggle Resume Dataset
- **Size:** 962 resumes
- **Categories:** 25 job categories
- **Format:** CSV with Resume text and Category columns

---

## ⚙️ Reproducing the Models

After cloning the repo and installing dependencies (`pip install -r requirements.txt`):

```bash
python notebooks/01_preprocessing.py   # clean & lemmatise the dataset
python notebooks/02_model.py            # train TF-IDF pipelines → models/resume_model.pkl
python notebooks/03_bert_model.py       # train BERT model       → models/bert_resume_model.pkl
streamlit run app.py
```

Run the test suite:

```bash
pytest tests/ -v
```

> **Note (data leakage fix):** `02_model.py` now wraps each classifier inside a
> `sklearn.Pipeline` so that the TF-IDF vectoriser is fitted **only on the
> training split**.  The pre-built `.pkl` files in the `models/` directory were
> trained with the old (leaky) code; retrain from scratch with the dataset to
> get correctly isolated models.

---


**Aman Pokhriyal**
- GitHub: https://github.com/amanwork5454-debug
- Live Project: https://aman-resume-screening.streamlit.app
