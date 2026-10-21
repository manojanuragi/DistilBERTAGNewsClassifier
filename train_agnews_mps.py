
#train_agnews_mps.py
import os
import time 
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import yaml 
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
def load_data(cfg, tokenizer):
  dataset = load_dataset("ag_news")

  def balance(split, n):
    per_class = n//4
    parts = [split.filter(lambda x, c=c : x["label"]== c).shuffle(seed=cfg["seed"]).select(range(per_class)) for c in range(4)]
    from datasets import concatenate_datasets

    return concatenate_datasets(parts).shuffle(seed=cfg["seed"])

  train_dataset = balance(dataset["train"], cfg["train_samples"])
  test_dataset = balance(dataset["test"], cfg["test_samples"])

  def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])

  train= train_dataset.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")
  test = test_dataset.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")
  return train, test


def main():
  with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
  
  torch.manual_seed(cfg["seed"])
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  torch.set_num_threads(os.cpu_count())
  print(f"Using device: {device} | threads: {torch.get_num_threads()}")

  #step 1 
  tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
  train_dataset, test_dataset = load_data(cfg, tokenizer)
  print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

  #step 2 : dataloader with workr process 
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], persistent_workers=bool(cfg["num_workers"]))
  test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=64, shuffle=False, num_workers=cfg["num_workers"])

  #step 3 : model +training
  model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=4, id2label=dict(enumerate(LABELS)), label2id={name: i for i , name in enumerate(LABELS)},).to(device)
  optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=len(train_dataloader) * cfg["epochs"],
  )
  start= time.time()
  for epoch in range(cfg["epochs"]):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg["epochs"]}"):
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      total_loss += loss.item()
    print(f"Epoch {epoch+1}/{cfg["epochs"]} | Loss: {total_loss/len(train_dataloader)}")
  print(f"Training time: {(time.time()-start)/60: 1f} min")

  #steop 4 : evalution 
  model.eval()
  y_true = []
  y_pred = []
  with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluation"):
      y_true.extend(batch.pop("labels").tolist())
      batch = {k: v.to(device) for k, v in batch.items()}
      y_pred.extend(model(**batch).logits.argmax(-1).tolist())

  os.makedirs(cfg["output_dir"], exist_ok=True)
  report = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
  print(report)
  print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
  with open(os.path.join(cfg["output_dir"], "report.txt"), "w") as f:
    f.write(report)
  
  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title("AG NEWS confusion metrix")
  plt.tight_layout()
  plt.savefig(os.path.join(cfg["output_dir"], "confusion_matrix.png"))
  plt.close()
  
  # save model 
  save_dir = f"{cfg["output_dir"]}/model-agnews"
  os.makedirs(save_dir, exist_ok=True)
  model.save_pretrained(save_dir)
  tokenizer.save_pretrained(save_dir)
if __name__ == "__main__":
  main()