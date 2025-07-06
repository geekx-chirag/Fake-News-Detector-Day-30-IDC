import logging
import torch
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Model configuration: Hugging Face repo names and ensemble weights
MODEL_CONFIG = [
    {"name": "raima2001/Fake-News-Detection-Roberta", "weight": 0.40},
    {"name": "vikram71198/distilroberta-base-finetuned-fake-news-detection", "weight": 0.35},
    {"name": "Pulk17/Fake-News-Detection", "weight": 0.25},
]

STD_LABELS = ["FAKE", "REAL"]

def load_model(repo: str):
    """Load HF text-classification pipeline on GPU if available, else CPU."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-classification", model=repo, tokenizer=repo, device=device)
    except Exception as e:
        logging.warning("⚠️ Could not load %s – %s", repo, e)
        return None

# Initialize pipelines
MODELS = []
for cfg in MODEL_CONFIG:
    model_pipeline = load_model(cfg["name"])
    if model_pipeline:
        MODELS.append({"pipeline": model_pipeline, "weight": cfg["weight"]})

if not MODELS:
    raise RuntimeError("No models loaded – check internet/HF credentials")

def normalise_label(raw_label: str) -> str:
    """Map various model labels explicitly to FAKE or REAL."""
    lbl = raw_label.upper()
    # Adjust these mappings if your models use different label conventions
    if lbl in {"LABEL_0", "FAKE", "NEGATIVE", "FALSE"}:
        return "FAKE"
    if lbl in {"LABEL_1", "REAL", "POSITIVE", "TRUE"}:
        return "REAL"
    print(f"Unknown label: {raw_label}")  # Debug
    return "REAL"  # Default fallback

def ensemble_predict(text: str):
    """Compute weighted soft-vote probabilities for FAKE vs REAL."""
    weighted_scores = {"FAKE": 0.0, "REAL": 0.0}
    for m in MODELS:
        # Get both class probabilities
        preds = m["pipeline"](text, truncation=True, max_length=512, return_all_scores=True)[0]
        for pred in preds:
            label = normalise_label(pred["label"])
            weighted_scores[label] += m["weight"] * float(pred["score"])
        print(f"Model: {m['pipeline'].model.name_or_path}, Raw Output: {preds}")  # Debug
    print(f"Weighted Scores: {weighted_scores}")  # Debug
    return weighted_scores

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    result_class = ""
    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if news_text:
            scores = ensemble_predict(news_text)
            fake_p, real_p = scores["FAKE"], scores["REAL"]
            if fake_p > real_p:
                result = "Likely Fake"
                result_class = "fake"
                confidence = f"{fake_p / (fake_p + real_p) * 100:.2f}"
            else:
                result = "Likely Real"
                result_class = "real"
                confidence = f"{real_p / (fake_p + real_p) * 100:.2f}"
    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           result_class=result_class)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
