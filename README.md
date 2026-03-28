\# 🤖 Resume Screening AI



An NLP-powered AI tool that automatically screens resumes and predicts the most suitable job category using Machine Learning.



🔗 \*\*Live Demo:\*\* https://aman-resume-screening.streamlit.app

📂 \*\*GitHub:\*\* https://github.com/amanwork5454-debug/RESUME-SCREENING



\---



\## 📌 Project Highlights

\- 📄 Upload PDF resume or paste text directly

\- 🎯 Predicts job category from 25 options instantly

\- 📊 Shows top 5 matching categories with confidence scores

\- 🤖 3 ML models compared — best achieves 100% accuracy

\- 🚀 Deployed live on Streamlit Cloud



\---



\## 🔬 How It Works

1\. \*\*Text Extraction\*\* — Reads PDF or accepts pasted text

2\. \*\*Text Cleaning\*\* — Removes URLs, symbols, special characters

3\. \*\*Stopword Removal\*\* — Removes common words

4\. \*\*Lemmatization\*\* — Reduces words to root form

5\. \*\*TF-IDF Vectorization\*\* — Converts text to 1500 numerical features

6\. \*\*ML Classification\*\* — Predicts job category from 25 options



\---



\## 🧠 ML Model Comparison


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

> Both TF-IDF and BERT achieve ~99.6% CV accuracy.
> BERT provides richer 768-dim semantic embeddings vs 1500 TF-IDF features.



\---



\## 🏷️ 25 Supported Job Categories

Data Science, Java Developer, Python Developer, DevOps Engineer,

Web Designing, HR, Testing, Blockchain, ETL Developer, Hadoop,

Sales, Mechanical Engineer, Database, Electrical Engineering,

Health and fitness, PMO, Business Analyst, DotNet Developer,

Automation Testing, Network Security Engineer, Civil Engineer,

SAP Developer, Advocate, Arts, Operations Manager



\---



\## 🛠️ Tech Stack

\- \*\*NLP:\*\* NLTK, TF-IDF Vectorization

\- \*\*ML:\*\* Scikit-learn (Random Forest, SVM, Logistic Regression)

\- \*\*PDF:\*\* PyPDF2

\- \*\*Dashboard:\*\* Streamlit

\- \*\*Deployment:\*\* Streamlit Cloud



\---



\## 📁 Project Structure

```

resume-screening/

├── app.py                      # Streamlit dashboard

├── requirements.txt

├── models/

│   └── resume\_model.pkl        # Trained model + vectorizer

├── notebooks/

│   ├── 01\_preprocessing.py     # Text cleaning \& NLP

│   ├── 02\_model.py             # Model training \& evaluation

│   ├── category\_distribution.png

│   ├── confusion\_matrix.png

│   └── model\_comparison.png

└── data/

&#x20;   └── UpdatedResumeDataSet.csv  # Not in GitHub (gitignored)

```



\## 👤 Author

\*\*Aman Pokhriyal\*\*

\- GitHub: https://github.com/amanwork5454-debug

