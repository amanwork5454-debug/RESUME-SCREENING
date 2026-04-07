"""Unit tests for utils.py."""
import pytest
from utils import clean_resume, lemmatize_text, extract_skills


# ── clean_resume ──────────────────────────────────────────────────────────────

class TestCleanResume:
    def test_lowercases(self):
        assert clean_resume("Python Developer") == "python developer"

    def test_removes_url(self):
        result = clean_resume("Visit https://github.com/user for more")
        assert "github" not in result
        assert "https" not in result

    def test_removes_hashtags(self):
        assert "#python" not in clean_resume("Skills: #python #ml")

    def test_removes_mentions(self):
        assert "@user" not in clean_resume("Contact @user on twitter")

    def test_collapses_whitespace(self):
        result = clean_resume("hello   world\t\tnewline")
        assert "  " not in result

    def test_removes_non_ascii(self):
        result = clean_resume("Résumé with café")
        assert "é" not in result

    def test_empty_string(self):
        assert clean_resume("") == ""

    def test_strips_whitespace(self):
        assert clean_resume("  hello  ") == "hello"


# ── lemmatize_text ────────────────────────────────────────────────────────────

class TestLemmatizeText:
    def test_removes_stop_words(self):
        result = lemmatize_text("this is a test of the system")
        assert "this" not in result.split()
        assert "is" not in result.split()
        assert "the" not in result.split()

    def test_keeps_meaningful_words(self):
        result = lemmatize_text("python developer machine learning")
        assert "python" in result
        assert "developer" in result

    def test_removes_short_tokens(self):
        # single-char and two-char words are stripped
        result = lemmatize_text("a to do it")
        tokens = result.split()
        assert all(len(t) > 2 for t in tokens)

    def test_empty_string(self):
        assert lemmatize_text("") == ""


# ── extract_skills ────────────────────────────────────────────────────────────

class TestExtractSkills:
    def test_detects_python(self):
        skills = extract_skills("Experienced Python developer")
        assert "Python" in skills.get("Languages", [])

    def test_detects_js_abbreviation(self):
        """'js' (word-boundary) should resolve to JavaScript."""
        skills = extract_skills("Frontend dev using js and React")
        assert "JavaScript" in skills.get("Languages", [])

    def test_detects_machine_learning_abbreviation(self):
        """'ml' should resolve to Machine Learning."""
        skills = extract_skills("5 years of ml experience")
        assert "Machine Learning" in skills.get("ML / AI", [])

    def test_detects_kubernetes_k8s(self):
        """'k8s' should resolve to Kubernetes."""
        skills = extract_skills("Deployed microservices on k8s clusters")
        assert "Kubernetes" in skills.get("Cloud / DevOps", [])

    def test_detects_multiple_domains(self):
        text = "Python, TensorFlow, AWS, React, SQL"
        skills = extract_skills(text)
        assert "Languages" in skills
        assert "ML / AI" in skills
        assert "Cloud / DevOps" in skills
        assert "Web / APIs" in skills
        assert "Data" in skills

    def test_no_false_positive_r_in_word(self):
        """'r' inside a word like 'developer' should NOT match the R language."""
        skills = extract_skills("experienced developer")
        langs = skills.get("Languages", [])
        assert "R" not in langs

    def test_empty_text_returns_empty(self):
        assert extract_skills("") == {}

    def test_no_duplicate_skills_from_aliases(self):
        """A skill matched via alias should appear only once, not twice."""
        skills = extract_skills("js javascript frontend")
        js_count = skills.get("Languages", []).count("JavaScript")
        assert js_count == 1

    def test_detects_sql(self):
        skills = extract_skills("Proficient in SQL and PostgreSQL")
        assert "SQL" in skills.get("Data", [])
        assert "PostgreSQL" in skills.get("Data", [])

    def test_detects_docker_and_kubernetes_full_name(self):
        skills = extract_skills("Docker and Kubernetes experience")
        cd = skills.get("Cloud / DevOps", [])
        assert "Docker" in cd
        assert "Kubernetes" in cd

    def test_detects_cpp(self):
        """'C++' should be detected as a C++ skill."""
        skills = extract_skills("Experienced C++ developer with 5 years")
        assert "C++" in skills.get("Languages", [])

    def test_no_false_positive_cpp_from_plain_c(self):
        """Text with only 'c' (no '++') should NOT match C++."""
        skills = extract_skills("machine learning accounting experienced")
        langs = skills.get("Languages", [])
        assert "C++" not in langs
