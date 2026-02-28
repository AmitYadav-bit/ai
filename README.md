# AI Text Analyzer

A lightweight, AI-powered text analysis tool built with Python. It provides **sentiment analysis**, **keyword extraction**, and **extractive text summarization** through a Flask REST API and a command-line interface.

## Features

| Feature | Description |
|---|---|
| **Sentiment Analysis** | Classifies text as *positive*, *negative*, or *neutral* with compound/positive/negative/neutral scores |
| **Keyword Extraction** | Extracts the most relevant keywords using TF-IDF scoring |
| **Text Summarization** | Returns an extractive summary keeping the most informative sentences |
| **REST API** | Flask-based HTTP API with JSON in/out |
| **CLI** | Command-line tool for quick analysis |

## Requirements

- Python 3.9+
- Dependencies are listed in `requirements.txt` and `pyproject.toml`

## Installation

```bash
pip install -e .
```

NLTK data (VADER lexicon, stopwords, punkt tokenizer) is downloaded automatically on first run.

## Usage

### Command-line

```bash
# Analyze text (all features)
echo "AI is transforming every industry!" | ai-analyzer analyze

# Sentiment only
ai-analyzer sentiment "I absolutely love this product!"

# Extract top 5 keywords
ai-analyzer keywords --top-n 5 "Deep learning models have revolutionized NLP tasks."

# Summarize (top 2 sentences)
ai-analyzer summarize --num-sentences 2 "$(cat article.txt)"

# Start the API server
ai-analyzer serve --port 5000
```

### REST API

Start the server:

```bash
ai-analyzer serve
```

**POST /analyze** — run all analyses

```bash
curl -s -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "AI is amazing and will change the world."}'
```

```json
{
  "sentiment": { "label": "positive", "compound": 0.78, "positive": 0.45, "negative": 0.0, "neutral": 0.55 },
  "keywords": ["ai", "amazing", "change", "world"],
  "summary": "AI is amazing and will change the world."
}
```

**POST /sentiment** — sentiment scores only

**POST /keywords** — keyword list (`top_n` parameter optional, default 10)

**POST /summarize** — extractive summary (`num_sentences` parameter optional, default 3)

**GET /health** — liveness probe

## Project Structure

```
ai/
├── src/
│   └── ai_analyzer/
│       ├── __init__.py   # package exports
│       ├── analyzer.py   # core AI logic
│       ├── api.py        # Flask REST API
│       └── cli.py        # CLI entry point
├── tests/
│   └── test_analyzer.py  # unit & API tests
├── pyproject.toml
└── requirements.txt
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## How It Works

- **Sentiment** — uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based model tuned for social-media and general text.
- **Keywords** — scores tokens by TF-IDF (term frequency × inverse document frequency estimated from sentence splits) and returns the top-N words.
- **Summarization** — scores each sentence by the sum of its normalised TF-IDF token weights and returns the top sentences in their original order.
