import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── Load Data ──
df = pd.read_csv('data/processed_resumes.csv')
print(f"Dataset shape: {df.shape}")

# ── Encode Labels ──
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])
print(f"Categories: {list(le.classes_)}")

# ── Features & Target (raw text — vectorizer fitted inside Pipeline) ──
X = df['processed_resume']
y = df['Category_encoded']

# ── Train/Test Split BEFORE any vectorization (prevents data leakage) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size:  {X_test.shape[0]}")

def _tfidf():
    return TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))

# ── Train 3 Pipelines (TF-IDF fitted only on training data) ──
pipelines = {
    'Logistic Regression': Pipeline([('tfidf', _tfidf()),
                                      ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'Random Forest':       Pipeline([('tfidf', _tfidf()),
                                      ('clf', RandomForestClassifier(n_estimators=200, random_state=42))]),
    'SVM':                 Pipeline([('tfidf', _tfidf()),
                                      ('clf', SVC(kernel='linear', probability=True, random_state=42))]),
}

results = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # CV only on training data — keeps test set truly held-out
    cv  = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    results[name] = {'pipeline': pipeline, 'accuracy': acc, 'cv': cv, 'y_pred': y_pred}
    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  CV Acc   : {cv:.4f} ({cv*100:.1f}%)")

# ── Best Model (selected by CV accuracy, not test accuracy) ──
best_name = max(results, key=lambda x: results[x]['cv'])
best = results[best_name]
print(f"\n✅ Best Model: {best_name} (CV Accuracy: {best['cv']*100:.1f}%)")

# ── Classification Report ──
print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, best['y_pred'],
      target_names=le.classes_))

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('notebooks/confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved")

# ── Model Comparison Chart ──
names = list(results.keys())
accs  = [results[n]['accuracy']*100 for n in names]
cvs   = [results[n]['cv']*100 for n in names]

x = np.arange(len(names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, accs, width, label='Test Accuracy', color='steelblue')
ax.bar(x + width/2, cvs,  width, label='CV Accuracy',   color='orange')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Comparison — Resume Screening')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.set_ylim(0, 110)
for i, (a, c) in enumerate(zip(accs, cvs)):
    ax.text(i - width/2, a + 1, f'{a:.1f}%', ha='center', fontsize=9)
    ax.text(i + width/2, c + 1, f'{c:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('notebooks/model_comparison.png')
plt.show()
print("✅ Model comparison chart saved")

# ── Save Best Pipeline ──
best_pipeline = best['pipeline']
model_data = {
    'model':      best_pipeline,                              # full Pipeline
    'tfidf':      best_pipeline.named_steps['tfidf'],        # for JD matching
    'le':         le,
    'model_name': best_name,
    'accuracy':   best['accuracy'],
    'cv':         best['cv'],
    'categories': list(le.classes_)
}
with open('models/resume_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✅ Model saved as models/resume_model.pkl")