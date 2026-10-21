
# DistilBERT AG News Classifier (CPU)

Fine-tunes `distilbert-base-uncased` on a balanced subset of AG News
(30,000 train / 3,000 test) to classify news into World, Sports, Business and Sci/Tech.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train_agnews_mps.py
```

## Outputs

- `outputs/distilbert-agnews/` – model.safetensors, config.json, tokenizer_config.json
- `outputs/classification_report.txt` – precision, recall, F1-score, accuracy
- `outputs/confusion_matrix.png` – confusion matrix chart


## Push to GitHub

```bash
git init
git add train_agnews_mps.py config.yaml requirements.txt README.md
git commit -m "DistilBERT AG News classifier"
git remote add origin https://github.com/your-username/agnews-distilbert.git
git push -u origin main
```