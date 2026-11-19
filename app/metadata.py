"""Load per-document metadata for the corpus.

Metadata comes from two places, merged in this order:
  1. The header block at the top of each .txt file: consecutive "Key: value"
     lines before the first blank or non-header line (e.g. Title, Source).
  2. An optional sidecar file, corpus/metadata.json, mapping
     filename -> {field: value}. Sidecar values override header values.

Header metadata is written by the document author, so it is untrusted just
like the document body. app.prompt escapes it the same way.
"""

import json
import re
from pathlib import Path

from app.retriever import load_corpus

HEADER_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]{0,31}):[ \t]*(\S.*)$")
VALID_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
RESERVED = {"name"}  # used for the filename in <document name="...">


def normalize_key(key):
    """'Published Date' -> 'published_date' (safe as an XML attribute name)."""
    return re.sub(r"[ -]+", "_", key.strip().lower())


def parse_header(text):
    """Split a document into (fields, body).

    Returns ({}, text) unchanged when the file has no header block.
    """
    lines = text.splitlines()
    fields = {}
    i = 0
    while i < len(lines):
        match = HEADER_LINE.match(lines[i].rstrip())
        if not match:
            break
        key = normalize_key(match.group(1))
        if key not in RESERVED:
            fields.setdefault(key, match.group(2).strip())
        i += 1
    if i == 0:
        return {}, text
    return fields, "\n".join(lines[i:]).lstrip("\n")


def load_sidecar(path, filenames):
    """Read and validate corpus/metadata.json. Missing file -> {}."""
    path = Path(path)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a JSON object of filename -> fields")

    result = {}
    for filename, fields in data.items():
        if filename not in filenames:
            raise ValueError(f"{path.name}: metadata for unknown document '{filename}'")
        if not isinstance(fields, dict):
            raise ValueError(f"{path.name}: fields for '{filename}' must be an object")
        clean = {}
        for raw_key, value in fields.items():
            key = normalize_key(raw_key)
            if not VALID_KEY.match(key) or key in RESERVED:
                raise ValueError(f"{path.name}: invalid key '{raw_key}' for '{filename}'")
            if isinstance(value, (dict, list)) or value is None:
                raise ValueError(f"{path.name}: '{raw_key}' for '{filename}' must be a scalar")
            clean[key] = str(value)
        result[filename] = clean
    return result


def load_metadata(corpus_dir, docs=None, sidecar="metadata.json"):
    """Return {filename: {field: value}} for every document in the corpus.

    Pass docs (from load_corpus) to avoid reading the files twice.
    """
    if docs is None:
        docs = load_corpus(corpus_dir)
    metadata = {name: parse_header(text)[0] for name, text in docs.items()}
    for name, fields in load_sidecar(Path(corpus_dir) / sidecar, docs).items():
        metadata[name].update(fields)
    return metadata
