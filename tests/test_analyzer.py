"""Tests for the AI Text Analyzer."""

import json
import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Artificial intelligence is transforming the world rapidly. "
    "Machine learning models can now understand human language with remarkable accuracy. "
    "Deep learning techniques have enabled breakthroughs in computer vision and NLP. "
    "However, there are important ethical considerations around AI deployment. "
    "Researchers continue to push the boundaries of what is possible with AI systems."
)

POSITIVE_TEXT = "I absolutely love this product! It is fantastic and amazing. Best purchase ever!"
NEGATIVE_TEXT = "This is terrible and awful. I hate it completely. Worst experience ever."
NEUTRAL_TEXT = "The box contains a book and a pen."


# ---------------------------------------------------------------------------
# TextAnalyzer tests
# ---------------------------------------------------------------------------

class TestTextAnalyzer:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ai_analyzer.analyzer import TextAnalyzer
        self.analyzer = TextAnalyzer()

    def test_analyze_returns_all_keys(self):
        result = self.analyzer.analyze(SAMPLE_TEXT)
        assert set(result.keys()) == {"sentiment", "keywords", "summary"}

    def test_analyze_empty_raises(self):
        with pytest.raises(ValueError):
            self.analyzer.analyze("")

    def test_analyze_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            self.analyzer.analyze("   ")

    # --- sentiment ---

    def test_sentiment_positive(self):
        result = self.analyzer.sentiment(POSITIVE_TEXT)
        assert result["label"] == "positive"
        assert result["compound"] > 0

    def test_sentiment_negative(self):
        result = self.analyzer.sentiment(NEGATIVE_TEXT)
        assert result["label"] == "negative"
        assert result["compound"] < 0

    def test_sentiment_has_expected_keys(self):
        result = self.analyzer.sentiment(NEUTRAL_TEXT)
        assert set(result.keys()) == {"positive", "negative", "neutral", "compound", "label"}

    def test_sentiment_scores_sum_to_one(self):
        result = self.analyzer.sentiment(SAMPLE_TEXT)
        total = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total - 1.0) < 0.01

    # --- keywords ---

    def test_keywords_returns_list(self):
        result = self.analyzer.keywords(SAMPLE_TEXT)
        assert isinstance(result, list)

    def test_keywords_top_n_respected(self):
        for n in (1, 3, 5):
            result = self.analyzer.keywords(SAMPLE_TEXT, top_n=n)
            assert len(result) <= n

    def test_keywords_no_stop_words(self):
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as sw
        stop = set(sw.words("english"))
        result = self.analyzer.keywords(SAMPLE_TEXT, top_n=10)
        for word in result:
            assert word not in stop

    # --- summarize ---

    def test_summarize_short_text_returned_as_is(self):
        short = "AI is great."
        result = self.analyzer.summarize(short, num_sentences=3)
        assert result.strip() == short.strip()

    def test_summarize_reduces_sentence_count(self):
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        result = self.analyzer.summarize(SAMPLE_TEXT, num_sentences=2)
        assert len(sent_tokenize(result)) <= 2

    def test_summarize_preserves_original_sentences(self):
        result = self.analyzer.summarize(SAMPLE_TEXT, num_sentences=2)
        # Each sentence in the summary must appear verbatim in the original
        import nltk
        from nltk.tokenize import sent_tokenize
        for sentence in sent_tokenize(result):
            assert sentence in SAMPLE_TEXT


# ---------------------------------------------------------------------------
# Flask API tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from ai_analyzer.api import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_analyze_success(self, client):
        resp = client.post("/analyze", json={"text": SAMPLE_TEXT})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sentiment" in data
        assert "keywords" in data
        assert "summary" in data

    def test_analyze_missing_text(self, client):
        resp = client.post("/analyze", json={})
        assert resp.status_code == 400

    def test_analyze_no_body(self, client):
        resp = client.post("/analyze")
        assert resp.status_code == 400

    def test_sentiment_endpoint(self, client):
        resp = client.post("/sentiment", json={"text": POSITIVE_TEXT})
        assert resp.status_code == 200
        assert resp.get_json()["label"] == "positive"

    def test_keywords_endpoint(self, client):
        resp = client.post("/keywords", json={"text": SAMPLE_TEXT, "top_n": 5})
        assert resp.status_code == 200
        assert len(resp.get_json()["keywords"]) <= 5

    def test_summarize_endpoint(self, client):
        resp = client.post("/summarize", json={"text": SAMPLE_TEXT, "num_sentences": 2})
        assert resp.status_code == 200
        assert "summary" in resp.get_json()

    def test_sentiment_missing_text(self, client):
        resp = client.post("/sentiment", json={})
        assert resp.status_code == 400

    def test_keywords_missing_text(self, client):
        resp = client.post("/keywords", json={})
        assert resp.status_code == 400

    def test_summarize_missing_text(self, client):
        resp = client.post("/summarize", json={})
        assert resp.status_code == 400
