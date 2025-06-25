import os
import boto3
import json
import pandas as pd
import datetime
from io import StringIO
from dotenv import load_dotenv

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

ECS_CLUSTER = os.getenv("ECS_CLUSTER")
ECS_TASK_DEFINITION = os.getenv("ECS_TASK_DEFINITION_ARN")  # Full ARN
SUBNET_ID = os.getenv("SUBNET_ID")
SECURITY_GROUP_ID = os.getenv("SECURITY_GROUP_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# ---------------------------
# Debug log
# ---------------------------
print("🐛 CONFIG VALUES:")
print(f"   ECS_CLUSTER         = {ECS_CLUSTER}")
print(f"   ECS_TASK_DEFINITION = {ECS_TASK_DEFINITION}")
print(f"   SUBNET_ID           = {SUBNET_ID}")
print(f"   SECURITY_GROUP_ID   = {SECURITY_GROUP_ID}")
print(f"   BUCKET_NAME         = {BUCKET_NAME}")

ecs_client = boto3.client("ecs", region_name="us-east-1")
s3 = boto3.client("s3")

# ---------------------------
# ECS Pipeline Trigger
# ---------------------------
def trigger_ecs_pipeline(filename: str, user: str):
    global ECS_TASK_DEFINITION
    print("✅ Inside trigger_ecs_pipeline", flush=True)
    print("   filename:", filename, "user:", user, flush=True)
    print("   ECS_TASK_DEFINITION =", ECS_TASK_DEFINITION, flush=True)

    def json_fallback(o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        return str(o)

    try:
        response = ecs_client.run_task(
            cluster=ECS_CLUSTER,
            launchType="FARGATE",
            taskDefinition=ECS_TASK_DEFINITION,
            count=1,
            platformVersion="LATEST",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": [SUBNET_ID],
                    "securityGroups": [SECURITY_GROUP_ID],
                    "assignPublicIp": "ENABLED"
                }
            },
            overrides={
                "containerOverrides": [
                    {
                        "name": "sentiment-pipeline",  # ECS container name
                        "environment": [
                            {"name": "BUCKET_NAME", "value": BUCKET_NAME},
                            {"name": "USER_NAME", "value": user},
                            {"name": "FILENAME", "value": filename}
                        ]
                    }
                ]
            }
        )

        print("🧾 ECS RunTask response:")
        print(json.dumps(response, indent=2, default=json_fallback))
        return response
    except Exception as e:
        print(f"❌ Failed to trigger ECS task: {e}")
        return None

# ---------------------------
# Dashboard Generation
# ---------------------------
def generate_dashboard(user: str, filename: str):
    output_key = f"output/{user}/served.csv"
    metadata_key = f"metadata/{user}/inference_summary.json"

    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=output_key)
        csv_str = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_str))

        summary = {
            "total_reviews": len(df),
            "positive_reviews": int((df["prediction"] == "positive").sum()),
            "negative_reviews": int((df["prediction"] == "negative").sum()),
            "neutral_reviews": int((df["prediction"] == "neutral").sum()) if "neutral" in df["prediction"].unique() else 0,
            "average_review_length": round(df["review"].astype(str).str.len().mean(), 2),
            "short_reviews_flagged": int(df.get("short_review", pd.Series([False] * len(df))).sum()),
            "top_positive_reviews": df[df["prediction"] == "positive"]["review"].head(5).tolist(),
            "top_negative_reviews": df[df["prediction"] == "negative"]["review"].head(5).tolist()
        }

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=metadata_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json"
        )

        print(f"✅ Dashboard summary uploaded to s3://{BUCKET_NAME}/{metadata_key}")

    except Exception as e:
        print(f"❌ Failed to generate dashboard: {e}")
