import os
import json
import boto3
import logging
import pandas as pd
from datetime import datetime
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.transformers

# -------------------------------------------
# Config
# -------------------------------------------
bucket = "mlops-sentiment-analysis-data"
train_s3 = f"s3://{bucket}/Silver/sampled.csv"
test_s3 = f"s3://{bucket}/Silver/test.csv"
metadata_key = "metadata/model_evaluation_summary.json"
tracking_uri = f"s3://{bucket}/mlflow"
experiment_name = "distilbert-sentiment"

local_train = "/tmp/train.csv"
local_test = "/tmp/test.csv"
model_output_dir = "/tmp/trained_model"
metadata_file = "/tmp/model_evaluation_summary.json"

# -------------------------------------------
# Logging Setup
# -------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------
# Download Data from S3
# -------------------------------------------
s3 = boto3.client("s3")

def download_from_s3(s3_uri, local_path):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3.download_file(bucket, key, local_path)
    logger.info(f"✅ Downloaded: {s3_uri} → {local_path}")

download_from_s3(train_s3, local_train)
download_from_s3(test_s3, local_test)

# -------------------------------------------
# Load Data
# -------------------------------------------
train_df = pd.read_csv(local_train).dropna()
test_df = pd.read_csv(local_test).dropna()
train_df = train_df.rename(columns={"review_body": "text", "label": "label"})
test_df = test_df.rename(columns={"review_body": "text", "label": "label"})

# -------------------------------------------
# Tokenization
# -------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------------------------
# Model Init
# -------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# -------------------------------------------
# Metrics
# -------------------------------------------
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

# -------------------------------------------
# TrainingArguments
# -------------------------------------------
training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# -------------------------------------------
# Train + Evaluate
# -------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"distilbert-{datetime.utcnow().isoformat()}"):
    logger.info("🚀 Starting training...")
    trainer.train()

    logger.info("🧪 Running evaluation...")
    metrics = trainer.evaluate()

    # -------------------------------------------
    # Calculate class counts
    true_labels = test_df["label"].tolist()
    predictions = trainer.predict(test_dataset).predictions.argmax(-1).tolist()

    actual_pos = sum(1 for l in true_labels if l == 1)
    actual_neg = sum(1 for l in true_labels if l == 0)
    pred_pos = sum(1 for l in predictions if l == 1)
    pred_neg = sum(1 for l in predictions if l == 0)

    test_df["prediction"] = predictions
    top_pos = test_df[test_df["prediction"] == 1].head(5)["text"].tolist()
    top_neg = test_df[test_df["prediction"] == 0].head(5)["text"].tolist()

    # -------------------------------------------
    # Log to MLflow
    mlflow.log_metrics(metrics)
    mlflow.log_metrics({
        "actual_positive": actual_pos,
        "actual_negative": actual_neg,
        "predicted_positive": pred_pos,
        "predicted_negative": pred_neg
    })

    with open("/tmp/top_reviews.json", "w") as f:
        json.dump({
            "positive_reviews": top_pos,
            "negative_reviews": top_neg
        }, f, indent=2)
    mlflow.log_artifact("/tmp/top_reviews.json")

    mlflow.transformers.log_model(
        transformers_model=model,
        artifact_path="distilbert_model",
        tokenizer=tokenizer,
        input_example={"text": "I love this product!"},
    )

    # -------------------------------------------
    # Save dashboard JSON to metadata
    eval_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": "distilbert-base-uncased",
        "metrics": metrics,
        "class_summary": {
            "actual_positive": actual_pos,
            "actual_negative": actual_neg,
            "predicted_positive": pred_pos,
            "predicted_negative": pred_neg
        },
        "sample_reviews": {
            "top_predicted_positive": top_pos,
            "top_predicted_negative": top_neg
        }
    }

    with open(metadata_file, "w") as f:
        json.dump(eval_summary, f, indent=4)

    s3.upload_file(
        Filename=metadata_file,
        Bucket=bucket,
        Key=metadata_key
    )
    logger.info(f"📤 Uploaded dashboard-ready JSON to: s3://{bucket}/{metadata_key}")
