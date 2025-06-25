import os
import json
import boto3
import pandas as pd
import tarfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- ENV VARIABLES ---
bucket = os.environ.get("BUCKET_NAME")
user = os.environ.get("USER_NAME")
filename = os.environ.get("FILENAME")

# --- S3 Keys ---
processed_key = f"processed/{user}/processed.csv"
output_dir_key = f"output/{user}/served.csv"
output_file_key = f"output/{user}/served.csv"
status_key = f"metadata/{user}/pipeline_status.json"
summary_key = f"metadata/{user}/inference_summary.json"
model_key = "models/model.tar.gz"

# --- S3 Client ---
s3 = boto3.client("s3")

# --- Step 1: Download processed data ---
print("📥 Downloading cleaned reviews...")
obj = s3.get_object(Bucket=bucket, Key=processed_key)
df = pd.read_csv(obj["Body"])
texts = df["review_clean"].astype(str).tolist()

# --- Step 2: Download and extract model ---
print("📦 Downloading model from S3...")
local_tar = "/tmp/model.tar.gz"
local_model_dir = "/tmp/model"

s3.download_file(bucket, model_key, local_tar)

print("📂 Extracting model...")
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(path=local_model_dir)

# --- Step 3: Load model + tokenizer from /tmp/model ---
print(f"🧠 Loading model from {local_model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(local_model_dir, local_files_only=True)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

# --- Step 4: Run inference ---
print("🚀 Running inference...")
preds = pipe(texts, truncation=True, padding=True)

# --- Step 5: Map predictions to labels ---
label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
df["prediction"] = [label_map.get(p["label"], "unknown") for p in preds]

# --- Step 6: Upload served.csv ---
tmp_csv_path = "/tmp/served.csv"
df.to_csv(tmp_csv_path, index=False)

s3.upload_file(tmp_csv_path, bucket, f"{output_dir_key}/part-00000.csv")
s3.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': f"{output_dir_key}/part-00000.csv"}, Key=output_file_key)

# Cleanup Spark-style folder
response = s3.list_objects_v2(Bucket=bucket, Prefix=output_dir_key + "/")
if "Contents" in response:
    s3.delete_objects(Bucket=bucket, Delete={'Objects': [{'Key': obj["Key"]} for obj in response["Contents"]]})

# --- Step 7: Upload dashboard summary ---
try:
    print("📊 Creating inference summary...")
    summary = {
        "total_reviews": len(df),
        "positive_reviews": int((df["prediction"] == "positive").sum()),
        "negative_reviews": int((df["prediction"] == "negative").sum()),
        "neutral_reviews": int((df["prediction"] == "neutral").sum()) if "neutral" in df["prediction"].unique() else 0,
        "average_review_length": round(df["review_clean"].astype(str).str.len().mean(), 2),
        "short_reviews_flagged": int(df.get("short_review", pd.Series([False] * len(df))).sum()),
        "top_positive_reviews": df[df["prediction"] == "positive"]["review_clean"].head(5).tolist(),
        "top_negative_reviews": df[df["prediction"] == "negative"]["review_clean"].head(5).tolist()
    }

    s3.put_object(
        Bucket=bucket,
        Key=summary_key,
        Body=json.dumps(summary, indent=2),
        ContentType="application/json"
    )
    print(f"✅ Dashboard summary saved to s3://{bucket}/{summary_key}")
except Exception as e:
    print(f"❌ Failed to create dashboard summary: {e}")

# --- Step 8: Mark pipeline as complete ---
s3.put_object(
    Bucket=bucket,
    Key=status_key,
    Body=json.dumps({"status": "inference_complete"}),
    ContentType="application/json"
)
print("✅ Inference complete.")
