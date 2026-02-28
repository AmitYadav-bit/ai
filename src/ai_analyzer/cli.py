"""Command-line interface for the AI Text Analyzer."""

import argparse
import json
import sys

from .analyzer import TextAnalyzer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-analyzer",
        description="AI-powered text analysis tool: sentiment, keywords, and summarization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze (all-in-one) ---
    p_analyze = subparsers.add_parser("analyze", help="Run all analyses.")
    p_analyze.add_argument("text", nargs="?", help="Text to analyze (or reads from stdin).")

    # --- sentiment ---
    p_sentiment = subparsers.add_parser("sentiment", help="Sentiment analysis.")
    p_sentiment.add_argument("text", nargs="?", help="Text to analyze (or reads from stdin).")

    # --- keywords ---
    p_keywords = subparsers.add_parser("keywords", help="Extract keywords.")
    p_keywords.add_argument("text", nargs="?", help="Text to analyze (or reads from stdin).")
    p_keywords.add_argument("--top-n", type=int, default=10, help="Number of keywords (default: 10).")

    # --- summarize ---
    p_summarize = subparsers.add_parser("summarize", help="Summarize text.")
    p_summarize.add_argument("text", nargs="?", help="Text to analyze (or reads from stdin).")
    p_summarize.add_argument(
        "--num-sentences", type=int, default=3, help="Number of sentences in summary (default: 3)."
    )

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Start the Flask API server.")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0).")
    p_serve.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000).")

    return parser


def _read_text(args) -> str:
    if args.command == "serve":
        return ""
    text = getattr(args, "text", None)
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Error: provide text as an argument or via stdin.", file=sys.stderr)
    sys.exit(1)


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        from .api import app  # noqa: PLC0415
        app.run(host=args.host, port=args.port, debug=False)
        return

    analyzer = TextAnalyzer()
    text = _read_text(args)

    if args.command == "analyze":
        result = analyzer.analyze(text)
    elif args.command == "sentiment":
        result = analyzer.sentiment(text)
    elif args.command == "keywords":
        result = {"keywords": analyzer.keywords(text, top_n=args.top_n)}
    elif args.command == "summarize":
        result = {"summary": analyzer.summarize(text, num_sentences=args.num_sentences)}
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
