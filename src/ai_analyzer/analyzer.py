"""Core AI text analysis module.

Provides sentiment analysis, keyword extraction, and extractive text
summarization using NLTK and scikit-learn.
"""

import math
import re
import string
from collections import Counter
from typing import Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data on first use
_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ("corpora/stopwords.zip", "stopwords"),
]


def _ensure_nltk_data() -> None:
    for path, resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


class TextAnalyzer:
    """Lightweight AI-powered text analyzer."""

    def __init__(self) -> None:
        _ensure_nltk_data()
        self._sia = SentimentIntensityAnalyzer()
        self._stop_words = set(stopwords.words("english"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Dict:
        """Run all analyses and return a combined result dict."""
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        return {
            "sentiment": self.sentiment(text),
            "keywords": self.keywords(text),
            "summary": self.summarize(text),
        }

    def sentiment(self, text: str) -> Dict:
        """Return VADER sentiment scores and a human-readable label.

        Returns a dict with keys: ``positive``, ``negative``,
        ``neutral``, ``compound``, and ``label``.
        """
        scores = self._sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {
            "positive": round(scores["pos"], 4),
            "negative": round(scores["neg"], 4),
            "neutral": round(scores["neu"], 4),
            "compound": round(compound, 4),
            "label": label,
        }

    def keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract the top *top_n* keywords using TF-IDF-style scoring.

        Words are ranked by term frequency weighted by inverse document
        frequency estimated from sentence-level splits.
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # Tokenize and normalize each sentence
        doc_freq: Counter = Counter()
        sentence_tokens: List[List[str]] = []
        for sentence in sentences:
            tokens = self._clean_tokens(sentence)
            sentence_tokens.append(tokens)
            doc_freq.update(set(tokens))  # count sentences containing word

        num_sentences = len(sentences)
        all_tokens: List[str] = [t for tokens in sentence_tokens for t in tokens]
        term_freq = Counter(all_tokens)

        if not term_freq:
            return []

        # TF * IDF scoring
        scores: Dict[str, float] = {}
        for word, count in term_freq.items():
            tf = count / len(all_tokens)
            idf = math.log((1 + num_sentences) / (1 + doc_freq[word])) + 1
            scores[word] = tf * idf

        sorted_words = sorted(scores, key=lambda w: scores[w], reverse=True)
        return sorted_words[:top_n]

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Return an extractive summary of *text*.

        Sentences are scored by the sum of TF-IDF scores of their tokens
        and the top *num_sentences* sentences are returned in original order.
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text.strip()

        # Build token frequency over the whole document
        all_tokens = self._clean_tokens(text)
        token_freq = Counter(all_tokens)
        max_freq = max(token_freq.values(), default=1)
        # Normalize frequencies
        norm_freq = {w: freq / max_freq for w, freq in token_freq.items()}

        # Score each sentence
        sentence_scores: Dict[int, float] = {}
        for idx, sentence in enumerate(sentences):
            tokens = self._clean_tokens(sentence)
            if tokens:
                sentence_scores[idx] = sum(norm_freq.get(t, 0) for t in tokens)

        # Pick the top-scoring sentence indices and sort by original position
        top_indices = sorted(
            sorted(sentence_scores, key=lambda i: sentence_scores[i], reverse=True)[
                :num_sentences
            ]
        )
        return " ".join(sentences[i] for i in top_indices)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_tokens(self, text: str) -> List[str]:
        """Lowercase, remove punctuation/numbers, filter stop words."""
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        return [t for t in tokens if t not in self._stop_words and len(t) > 1]
