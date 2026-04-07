import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import time

# ── Load Data ──
df = pd.read_csv('data/processed_resumes.csv')
print(f"Dataset shape: {df.shape}")

# ── Encode Labels ──
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])
print(f"Categories: {len(le.classes_)}")

# ── Load Sentence Transformer ──
print("🔄 Loading all-MiniLM-L6-v2 (sentence-transformers)...")
print("(This will download ~90MB on first run)")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ all-MiniLM-L6-v2 loaded!")

# ── Generate BERT Embeddings ──
print("\n🔄 Generating embeddings for all resumes...")
print("(This may take 5-10 minutes on CPU...)")
start = time.time()

# Use cleaned resume text
texts = df['processed_resume'].tolist()
embeddings = bert_model.encode(texts,
                                batch_size=16,
                                show_progress_bar=True)
end = time.time()
print(f"✅ Embeddings generated in {end-start:.0f} seconds")
print(f"Embedding shape: {embeddings.shape}")

# ── Save embeddings to avoid recomputing ──
np.save('data/bert_embeddings.npy', embeddings)
print("✅ Embeddings saved")

# ── Features & Target ──
X = embeddings
y = df['Category_encoded']

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Train Models on BERT Embeddings ──
models = {
    'Logistic Regression + BERT': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest + BERT':       RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # CV only on training data — keeps test set truly held-out
    cv  = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    results[name] = {'model': model, 'accuracy': acc, 'cv': cv}
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"  CV Acc   : {cv*100:.1f}%")

# ── Best Model (selected by CV accuracy) ──
best_name = max(results, key=lambda x: results[x]['cv'])
best = results[best_name]
print(f"\n✅ Best BERT Model: {best_name}")
print(f"   Accuracy: {best['accuracy']*100:.1f}%")
print(f"   CV Acc  : {best['cv']*100:.1f}%")

# ── Comparison Chart: TF-IDF vs BERT ──
categories_compare = ['TF-IDF\nRandom Forest', 'TF-IDF\nLogistic Reg',
                       'BERT\nLogistic Reg', 'BERT\nRandom Forest']
test_accs = [100.0, 99.5,
             results['Logistic Regression + BERT']['accuracy']*100,
             results['Random Forest + BERT']['accuracy']*100]
cv_accs   = [99.6, 99.3,
             results['Logistic Regression + BERT']['cv']*100,
             results['Random Forest + BERT']['cv']*100]

x = np.arange(len(categories_compare))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, test_accs, width,
               label='Test Accuracy', color=['#3b82f6','#3b82f6','#8b5cf6','#8b5cf6'])
bars2 = ax.bar(x + width/2, cv_accs,  width,
               label='CV Accuracy',  color=['#60a5fa','#60a5fa','#a78bfa','#a78bfa'])
ax.set_ylabel('Accuracy (%)')
ax.set_title('TF-IDF vs BERT — Resume Screening Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories_compare)
ax.legend()
ax.set_ylim(85, 105)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('notebooks/tfidf_vs_bert.png')
plt.show()
print("✅ Comparison chart saved")

# ── Save BERT Model ──
bert_model_data = {
    'classifier':  best['model'],
    'bert_model':  'all-MiniLM-L6-v2',
    'le':          le,
    'model_name':  best_name,
    'accuracy':    best['accuracy'],
    'cv':          best['cv'],
    'categories':  list(le.classes_)
}
with open('models/bert_resume_model.pkl', 'wb') as f:
    pickle.dump(bert_model_data, f)
print("✅ BERT model saved!")

print("\n" + "="*50)
print("FINAL COMPARISON SUMMARY")
print("="*50)
print(f"TF-IDF + Random Forest : 100.0% test | 99.6% CV")
print(f"BERT  + {best_name.split('+')[0]}: {best['accuracy']*100:.1f}% test | {best['cv']*100:.1f}% CV")
print("="*50)