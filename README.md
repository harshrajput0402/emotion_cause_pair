# Emotion–Cause Pair Extraction

A deep-learning web app that reads English text and finds **which clauses express an emotion** and **which clauses explain why** — then pairs them up.

> *"I was so happy, because I finally passed the exam."*
> → **Emotion clause:** "I was so happy" · **Cause clause:** "because I finally passed the exam"

<!-- Add a screenshot or GIF of the app here, e.g.: -->
<!-- ![Demo](docs/demo.gif) -->
---

## Features

- **Clause-level classification** – every clause is scored for being an *emotion* clause and/or a *cause* clause.
- **Emotion–cause pairing** – candidate pairs are ranked by `emotion_prob × cause_prob`.
- **Adjustable sensitivity threshold** – tune precision vs. recall live from the UI.
- **Unknown-word reporting** – shows which words were outside the model's vocabulary.
- **REST API** – `POST /api/predict` and `GET /api/health`.
- **Containerised** – one `Dockerfile`, runs anywhere.

## How it works

```
Text ─► sentence split ─► clause split ─► clean + pad ─► word IDs
                                                           │
                                          Embedding (200-d) + self-attention
                                                           │
                                  ┌────────────────────────┴───────────────┐
                           BiLSTM(100) → Dense                    BiLSTM(100) → Dense
                           → sigmoid (emotion)                    → sigmoid (cause)
                                  └────────────────┬───────────────────────┘
                                         emotion × cause score
                                                   ▼
                                       ranked emotion–cause pairs
```

1. **Preprocessing** (`preprocess.py`) – the corpus is annotated with emotion tags and `<cause>` spans. Sentences are split into clauses on punctuation, cleaned, padded, and converted to vocabulary IDs (unknown words map to `*UNK*`).
2. **Multi-task model** (`model.py`, class `ECModel`) – a shared embedding + attention layer feeds two parallel heads (BiLSTM → Dense → sigmoid), one for emotion clauses and one for cause clauses. Loss is a weighted sum of the two binary cross-entropies.
3. **Pair filter** (`FilterModel`) – a small dense network that classifies emotion/cause clause-embedding pairs as valid or not, using class weighting to handle the rarity of true pairs.
4. **Serving** (`app.py`) – Flask loads the saved checkpoint at start-up and exposes the prediction API and the web UI (`index.html`).

## Dataset

`data.txt` contains ~820 annotated sentences, each tagged with one of seven emotions (happy, anger, fear, sad, shame, surprise, disgust) and with the cause span marked. Data is split **90 % train / 10 % test**.

## Project structure

```
├── app.py            # Flask backend + model loading
├── model.py          # ECModel, FilterModel, training & evaluation
├── preprocess.py     # Parsing, clause splitting, vocab, padding
├── index.html        # Front-end UI
├── data.txt          # Annotated corpus
├── checkpoints/      # Trained model weights
├── Dockerfile
└── requirements.txt
```

## Getting started

### Run with Docker

```bash
docker build -t emotion-cause .
docker run -p 7860:7860 emotion-cause
```

Open <http://localhost:7860>.

### Run locally

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

> `app.py` reads the port from its own configuration; if you run it directly, see the console for the URL (default Flask port is 5000).

### Re-train the model

```bash
python model.py --mode train   # trains for 10 epochs and saves to ./checkpoints
python model.py --mode test    # evaluates the saved checkpoint
```

Evaluation reports precision, recall and F1.

## API

**`GET /api/health`**

```json
{ "model_loaded": true, "vocab_size": 2500 }
```

**`POST /api/predict`**

```json
{ "text": "I was so happy, because I finally passed the exam.", "threshold": 0.5 }
```

Returns, for each sentence: per-clause emotion/cause probabilities, boolean flags, unknown words, and a ranked list of emotion–cause pairs.
Limits: 5000 characters of input, first 20 sentences processed.

## Limitations

- Small training corpus → works best on simple, everyday English sentences.
- Each clause is classified independently, so long-range context across sentences is not used.
- Words not seen in training are treated as unknown.

## Possible improvements

- Replace the learned embeddings with pre-trained ones (GloVe) or a transformer encoder (BERT).
- Use clause-position and context features when pairing.
- Train on a larger corpus and report benchmark results.

## Tech stack

Python · TensorFlow / Keras · NumPy · Flask · Gunicorn · Docker · HTML/CSS/JavaScript
