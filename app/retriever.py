"""Small, deterministic document retriever for the D2 testing harness.

The corpus contains only the team's five synthetic .txt files. This module
ranks documents by unique keyword overlap; it does not interpret or execute
instructions found in document text. No API, database, or package is needed.
"""

import re
from pathlib import Path

STOPWORDS = {
    "a", "an", "the", "is", "are", "to", "of", "in", "on", "at", "for", "and", "or",
    "do", "does", "i", "my", "can", "what", "when", "how", "which", "it", "be", "go",
}


def tokenize(text):
    """Extract lowercase words, excluding common question/grammar words."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def load_corpus(corpus_dir):
    """Load the synthetic UTF-8 .txt corpus as {filename: full_text}.

    File ordering is stable across machines. Fail early rather than silently
    running experiments with a missing or empty corpus.
    """
    directory = Path(corpus_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")

    paths = sorted(p for p in directory.glob("*.txt") if p.is_file())
    if not paths:
        raise ValueError(f"no .txt documents in corpus directory: {directory}")

    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def retrieve(query, docs, k=2):
    """Return up to k (filename, text) pairs with matching query keywords.

    Score = number of *distinct* query keywords found in a document. Counting
    each word only once prevents a repeated word from unfairly dominating the
    ranking. Ties are resolved by filename, making retrieval reproducible.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    ranked = []
    for filename, text in docs.items():
        score = len(query_terms.intersection(tokenize(text)))
        if score:
            ranked.append((score, filename))

    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    return [(filename, docs[filename]) for _, filename in ranked[:k]]
