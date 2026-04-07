"""
Shared utility functions for resume cleaning, processing, and skill extraction.
Imported by both app.py and tests/.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Skill vocabulary (canonical display name → list of match tokens) ──────────
SKILLS = {
    "Languages": {
        "Python":       ["python"],
        "JavaScript":   ["javascript", r"\bjs\b"],
        "TypeScript":   ["typescript", r"\bts\b"],
        "Java":         ["java"],
        "C++":          ["c++"],
        "C#":           ["c#"],
        "Scala":        ["scala"],
        "Kotlin":       ["kotlin"],
        "Swift":        ["swift"],
        "Go":           [r"\bgo\b", r"\bgolang\b"],
        "Rust":         ["rust"],
        "PHP":          [r"\bphp\b"],
        "Ruby":         ["ruby"],
        "R":            [r"\br\b"],
        "MATLAB":       ["matlab"],
        "Julia":        ["julia"],
    },
    "ML / AI": {
        "Machine Learning": ["machine learning", r"\bml\b"],
        "Deep Learning":    ["deep learning", r"\bdl\b"],
        "NLP":              [r"\bnlp\b", "natural language processing"],
        "Computer Vision":  ["computer vision", r"\bcv\b"],
        "TensorFlow":       ["tensorflow"],
        "PyTorch":          ["pytorch"],
        "Keras":            ["keras"],
        "scikit-learn":     ["scikit-learn", "sklearn"],
        "BERT":             [r"\bbert\b"],
        "Transformers":     ["transformers"],
        "XGBoost":          ["xgboost"],
        "LightGBM":         ["lightgbm"],
        "LLM":              [r"\bllm\b", "large language model"],
        "Reinforcement Learning": ["reinforcement learning", r"\brl\b"],
    },
    "Data": {
        "SQL":          [r"\bsql\b"],
        "Pandas":       ["pandas"],
        "NumPy":        ["numpy"],
        "Spark":        [r"\bspark\b"],
        "Hadoop":       ["hadoop"],
        "Kafka":        ["kafka"],
        "Tableau":      ["tableau"],
        "Power BI":     ["power bi"],
        "PostgreSQL":   ["postgresql", "postgres"],
        "MySQL":        ["mysql"],
        "MongoDB":      ["mongodb"],
        "Airflow":      ["airflow"],
        "dbt":          [r"\bdbt\b"],
    },
    "Cloud / DevOps": {
        "AWS":          [r"\baws\b", "amazon web services"],
        "Azure":        ["azure"],
        "GCP":          [r"\bgcp\b", "google cloud"],
        "Docker":       ["docker"],
        "Kubernetes":   ["kubernetes", r"\bk8s\b"],
        "Git":          [r"\bgit\b"],
        "Linux":        ["linux"],
        "Terraform":    ["terraform"],
        "Jenkins":      ["jenkins"],
        "CI/CD":        [r"\bci/cd\b", r"\bcicd\b"],
    },
    "Web / APIs": {
        "React":        ["react", r"\breactjs\b"],
        "Angular":      ["angular"],
        "Vue":          [r"\bvue\b", r"\bvuejs\b"],
        "Django":       ["django"],
        "Flask":        ["flask"],
        "FastAPI":      ["fastapi"],
        "Node.js":      ["node.js", r"\bnodejs\b", r"\bnode\b"],
        "REST API":     ["rest api", r"\brestful\b"],
        "GraphQL":      ["graphql"],
        "HTML":         [r"\bhtml\b"],
        "CSS":          [r"\bcss\b"],
    },
}

# Domain → CSS badge class mapping (consumed by app.py)
DOMAIN_BADGE_CLASS = {
    "Languages":      "badge-languages",
    "ML / AI":        "badge-ml",
    "Data":           "badge-data",
    "Cloud / DevOps": "badge-cloud",
    "Web / APIs":     "badge-web",
}


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_resume(text: str) -> str:
    """Remove URLs, social noise, punctuation, and non-ASCII; lowercase."""
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def lemmatize_text(text: str) -> str:
    """Remove stop-words and very short tokens."""
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)


# ── Skill extraction ───────────────────────────────────────────────────────────

def extract_skills(text: str) -> dict[str, list[str]]:
    """
    Return {domain: [canonical_skill_name, ...]} for all skills found in *text*.

    Matching is done via regex so abbreviations like 'js', 'k8s', 'ml' are
    detected even when surrounded by word boundaries.
    """
    text_lower = text.lower()
    found: dict[str, list[str]] = {}
    for domain, skill_map in SKILLS.items():
        matched: list[str] = []
        for canonical, patterns in skill_map.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    matched.append(canonical)
                    break  # don't double-count via alias
        if matched:
            found[domain] = matched
    return found
