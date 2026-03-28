import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ── Load Data ──
df = pd.read_csv('data/UpdatedResumeDataSet.csv')
print(f"Dataset shape: {df.shape}")
print(f"Categories: {df['Category'].nunique()}")
print(f"\nCategory Distribution:")
print(df['Category'].value_counts())

# ── Text Cleaning Function ──
def clean_resume(text):
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove RT and cc
    text = re.sub(r'RT|cc', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\S+', ' ', text)
    # Remove mentions
    text = re.sub(r'@\S+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove non-ASCII
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Lowercase
    text = text.lower().strip()
    return text

# ── Apply Cleaning ──
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# ── Lemmatization ──
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

df['processed_resume'] = df['cleaned_resume'].apply(lemmatize_text)

print("\nSample cleaned resume:")
print(df['processed_resume'][0][:300])

# ── Category Distribution Chart ──
plt.figure(figsize=(12, 8))
df['Category'].value_counts().plot(kind='barh', color='steelblue')
plt.title('Resume Count by Job Category')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('notebooks/category_distribution.png')
plt.show()
print("✅ Category distribution chart saved")

# ── Save Processed Data ──
df.to_csv('data/processed_resumes.csv', index=False)
print("✅ Processed data saved")
print(f"\nFinal shape: {df.shape}")