# 🤖 Resume Screening AI

An NLP-powered AI tool that automatically screens resumes and predicts the most suitable job category using Machine Learning and Transformer models (DistilBERT).

🔗 **Live Demo:** https://aman-resume-screening.streamlit.app
📂 **GitHub:** https://github.com/amanwork5454-debug/RESUME-SCREENING

---

## 📌 Project Highlights
- 📄 Upload PDF resume or paste text directly
- 🎯 Predicts job category from 25 options instantly
- 📊 Shows top 5 matching categories with confidence scores
- 🤖 3 ML models compared (TF-IDF based)
- 🧠 DistilBERT transformer model implemented & compared
- 🚀 Deployed live on Streamlit Cloud

---

## 🔬 How It Works
1. **Text Extraction** — Reads PDF or accepts pasted text
2. **Text Cleaning** — Removes URLs, symbols, special characters
3. **Stopword Removal** — Removes common words like 'the', 'is'
4. **Lemmatization** — Reduces words to root form
5. **TF-IDF Vectorization** — Converts text to 1500 numerical features
6. **ML Classification** — Predicts job category from 25 options

---

## 🧠 ML Model Comparison

### TF-IDF Based Models
| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| **Random Forest** | **100%** | **99.6%** |
| Logistic Regression | 99.5% | 99.3% |
| SVM | 99.5% | 99.6% |

### BERT Based Models (DistilBERT)
| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| Logistic Regression + BERT | 99.5% | 99.6% |
| Random Forest + BERT | 99.5% | 99.6% |

> **Key Insight:** Both TF-IDF and BERT achieve ~99.6% CV accuracy.
> BERT provides richer 768-dimensional semantic embeddings vs
> 1500 TF-IDF features and generalizes better on unseen resume formats.

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
| NLTK | Text preprocessing, lemmatization |
| TF-IDF | Text vectorization (1500 features) |
| Scikit-learn | ML models, evaluation, cross validation |
| HuggingFace | DistilBERT transformer model |
| Sentence-Transformers | BERT embeddings (768 dimensions) |
| PyPDF2 | PDF text extraction |
| Streamlit | Web dashboard |
| Streamlit Cloud | Deployment |
| Git, GitHub | Version control |

---

## 📁 Project Structure
```
resume-screening/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── models/
│   ├── resume_model.pkl            # TF-IDF + Random Forest model
│   └── bert_resume_model.pkl       # DistilBERT + Logistic Regression
├── notebooks/
│   ├── 01_preprocessing.py         # Text cleaning & NLP pipeline
│   ├── 02_model.py                 # TF-IDF model training & evaluation
│   ├── 03_bert_model.py            # DistilBERT training & comparison
│   ├── category_distribution.png   # Resume count by category
│   ├── confusion_matrix.png        # Model confusion matrix
│   ├── model_comparison.png        # TF-IDF models comparison
│   └── tfidf_vs_bert.png           # TF-IDF vs BERT comparison
└── data/
    └── UpdatedResumeDataSet.csv    # Not in GitHub (gitignored)
```

## 📊 Dataset
- **Source:** Kaggle Resume Dataset
- **Size:** 962 resumes
- **Categories:** 25 job categories
- **Format:** CSV with Resume text and Category columns

---

## 👤 Author
**Aman Pokhriyal**
- GitHub: https://github.com/amanwork5454-debug
- Live Project: https://aman-resume-screening.streamlit.app