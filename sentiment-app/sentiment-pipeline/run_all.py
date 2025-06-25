import os
import boto3
import json
import subprocess

# -------------------------------------
# Environment
# -------------------------------------
bucket = os.environ.get("BUCKET_NAME")
user = os.environ.get("USER_NAME")
status_key = f"metadata/{user}/pipeline_status.json"

s3 = boto3.client("s3")

# -------------------------------------
# Update status to running
# -------------------------------------
s3.put_object(
    Bucket=bucket,
    Key=status_key,
    Body=json.dumps({"status": "running"}),
    ContentType="application/json"
)
print("🚀 Pipeline status set to: running")

# -------------------------------------
# Run Clean + Inference
# -------------------------------------
try:
    print("🧼 Running run_clean.py ...")
    subprocess.run(["python", "run_clean.py"], check=True)

    print("🧠 Running run_infer.py ...")
    subprocess.run(["python", "run_infer.py"], check=True)

except subprocess.CalledProcessError as e:
    print(f"❌ Pipeline failed: {e}")
    s3.put_object(
        Bucket=bucket,
        Key=status_key,
        Body=json.dumps({"status": "failed", "error": str(e)}),
        ContentType="application/json"
    )
    raise

print("✅ Full pipeline completed.")
