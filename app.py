"""
Backend for the Emotion-Cause demo.

Run:   python app.py
Open:  http://127.0.0.1:5000

Needs: model.py, preprocess.py, data.txt and the ./checkpoints folder
(created by `python model.py --mode train`) in the same folder as this file.
"""
import os
import re
import sys
from importlib import import_module

# model.py parses command-line arguments when imported, so give it a clean argv.
sys.argv = [sys.argv[0]]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
try:
    tf = import_module("tensorflow")
except ImportError as exc:
    raise RuntimeError(
        "TensorFlow is required to run this app. Install it with `python -m pip install tensorflow`."
    ) from exc

try:
    flask = import_module("flask")
except ImportError as exc:
    raise RuntimeError(
        "Flask is required to run this app. Install it with `python -m pip install flask`."
    ) from exc

Flask = flask.Flask
jsonify = flask.jsonify
request = flask.request
send_file = flask.send_file

import model as M
from preprocess import get_data, clean_clause, pad_corpus, convert_clauses_to_id

# ---------------------------------------------------------------- load model
print("Loading vocabulary from data.txt ...")
(_, _, _, _, _, _, _, _, word2id, pad_index, clause_size) = get_data("data.txt")

ec_model = M.ECModel(len(word2id), clause_size)
ec_model.call(np.zeros((1, clause_size), dtype="int32"))  # build the weights

checkpoint = tf.train.Checkpoint(ec_extract_model=ec_model)
manager = tf.train.CheckpointManager(checkpoint, "./checkpoints", max_to_keep=3)
MODEL_LOADED = manager.latest_checkpoint is not None
if MODEL_LOADED:
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print("Loaded trained weights from", manager.latest_checkpoint)
else:
    print("WARNING: no checkpoint found. Run `python model.py --mode train` first.")


# ------------------------------------------------------------ text handling
def split_sentences(text):
    """Split the input into sentences (new lines or . ! ? followed by a space)."""
    sentences = []
    for line in text.splitlines():
        for part in re.split(r"(?<=[.!?])\s+", line.strip()):
            if part.strip():
                sentences.append(part.strip())
    return sentences


def split_clauses(sentence):
    """Same clause splitting and cleaning as preprocess.py (plus '?')."""
    clauses = []
    for raw in re.split(r"[.,!?;:\"]+", sentence):
        words = clean_clause(raw).split()
        if words:
            clauses.append((raw.strip(), words))
    return clauses


# --------------------------------------------------------------------- app
app = Flask(__name__)


@app.get("/")
def index():
    return send_file("index.html")


@app.get("/api/health")
def health():
    return jsonify(model_loaded=MODEL_LOADED, vocab_size=len(word2id))


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    try:
        threshold = float(body.get("threshold", 0.5))
    except (TypeError, ValueError):
        threshold = 0.5

    if not text:
        return jsonify(error="Please type some text first."), 400
    if len(text) > 5000:
        return jsonify(error="Text is too long (max 5000 characters)."), 400

    results = []
    for sentence in split_sentences(text)[:20]:
        clauses = split_clauses(sentence)
        if not clauses:
            continue

        word_lists = [words[: clause_size - 1] for _, words in clauses]
        padded = pad_corpus([list(w) for w in word_lists], clause_size)
        ids = convert_clauses_to_id(word2id, padded)

        emotion_probs, cause_probs = ec_model.call(ids)
        emotion_probs = emotion_probs.numpy().reshape(-1)
        cause_probs = cause_probs.numpy().reshape(-1)

        out_clauses = []
        for i, (raw, words) in enumerate(clauses):
            out_clauses.append({
                "text": raw,
                "emotion_prob": round(float(emotion_probs[i]), 4),
                "cause_prob": round(float(cause_probs[i]), 4),
                "is_emotion": bool(emotion_probs[i] > threshold),
                "is_cause": bool(cause_probs[i] > threshold),
                "unknown_words": [w for w in words if w not in word2id],
            })

        # Candidate pairs: every emotion clause combined with every cause clause.
        pairs = []
        for e, ec in enumerate(out_clauses):
            for c, cc in enumerate(out_clauses):
                if ec["is_emotion"] and cc["is_cause"] and e != c:
                    pairs.append({
                        "emotion": ec["text"],
                        "cause": cc["text"],
                        "score": round(ec["emotion_prob"] * cc["cause_prob"], 4),
                    })
        pairs.sort(key=lambda p: p["score"], reverse=True)

        results.append({"sentence": sentence, "clauses": out_clauses, "pairs": pairs})

    if not results:
        return jsonify(error="No words found in the text."), 400

    return jsonify(model_loaded=MODEL_LOADED, threshold=threshold, results=results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
