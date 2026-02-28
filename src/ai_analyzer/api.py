"""Flask REST API for the AI Text Analyzer."""

import json
from http import HTTPStatus

from flask import Flask, Response, jsonify, request

from .analyzer import TextAnalyzer

app = Flask(__name__)
_analyzer = TextAnalyzer()


@app.route("/health", methods=["GET"])
def health() -> Response:
    """Simple liveness probe."""
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze() -> Response:
    """Run all analyses (sentiment + keywords + summary).

    Request JSON: ``{"text": "<input text>"}``

    Response JSON::

        {
            "sentiment": {"label": "positive", "compound": 0.8, ...},
            "keywords":  ["ai", "model", ...],
            "summary":   "..."
        }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), HTTPStatus.BAD_REQUEST

    try:
        result = _analyzer.analyze(str(data["text"]))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.UNPROCESSABLE_ENTITY

    return jsonify(result)


@app.route("/sentiment", methods=["POST"])
def sentiment() -> Response:
    """Return sentiment scores for the given text.

    Request JSON: ``{"text": "<input text>"}``
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), HTTPStatus.BAD_REQUEST

    try:
        result = _analyzer.sentiment(str(data["text"]))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.UNPROCESSABLE_ENTITY

    return jsonify(result)


@app.route("/keywords", methods=["POST"])
def keywords() -> Response:
    """Return top keywords for the given text.

    Request JSON: ``{"text": "<input text>", "top_n": 10}``
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), HTTPStatus.BAD_REQUEST

    top_n = int(data.get("top_n", 10))
    try:
        result = _analyzer.keywords(str(data["text"]), top_n=top_n)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.UNPROCESSABLE_ENTITY

    return jsonify({"keywords": result})


@app.route("/summarize", methods=["POST"])
def summarize() -> Response:
    """Return an extractive summary of the given text.

    Request JSON: ``{"text": "<input text>", "num_sentences": 3}``
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), HTTPStatus.BAD_REQUEST

    num_sentences = int(data.get("num_sentences", 3))
    try:
        result = _analyzer.summarize(str(data["text"]), num_sentences=num_sentences)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.UNPROCESSABLE_ENTITY

    return jsonify({"summary": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
