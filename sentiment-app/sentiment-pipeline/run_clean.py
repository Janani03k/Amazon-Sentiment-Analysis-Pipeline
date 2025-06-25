import os
import json
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, length, udf
from pyspark.sql.types import BooleanType

# --- ENV VARIABLES ---
bucket = os.environ.get("BUCKET_NAME", "mlops-sentiment-app")
user = os.environ.get("USER_NAME", "default")
filename = os.environ.get("FILENAME", "test.csv")

raw_key = f"uploads/raw/{user}/{filename}"
processed_dir_key = f"processed/{user}/processed.csv"
processed_file_key = f"processed/{user}/processed.csv"
summary_key = f"metadata/{user}/preprocessing_summary.json"
status_key = f"metadata/{user}/pipeline_status.json"

input_path = f"s3a://{bucket}/{raw_key}"
output_dir = f"s3a://{bucket}/{processed_dir_key}"

# --- AWS S3 Client ---
s3 = boto3.client("s3")

# --- Helper to Mark Failure ---
def mark_failed():
    s3.put_object(
        Bucket=bucket,
        Key=status_key,
        Body=json.dumps({"status": "failed"}),
        ContentType="application/json"
    )
    print("❌ Pipeline status marked as 'failed'")

try:
    # --- Start Spark ---
    spark = SparkSession.builder.appName("SentimentCleaner").getOrCreate()
    spark._jsc.hadoopConfiguration().set(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
    )

    # --- Load CSV ---
    df = spark.read.option("header", True).csv(input_path)
    original_count = df.count()

    # --- Clean & Transform ---
    review_col = "review_body" if "review_body" in df.columns else "review"
    if review_col not in df.columns:
        raise Exception("❌ No review column found.")

    df = df.withColumn(review_col, trim(col(review_col)))
    df = df.withColumn("review_clean", lower(col(review_col)))
    df_filtered = df.filter(col("review_clean").isNotNull() & (length(col("review_clean")) > 0))
    null_dropped = original_count - df_filtered.count()
    df_filtered = df_filtered.filter(~col("review_clean").rlike("^[0-9\\s]+$"))
    numeric_dropped = original_count - null_dropped - df_filtered.count()

    def is_short(text):
        if not text:
            return True
        words = len(text.split())
        chars = len(text)
        return words < 5 or chars < 30

    is_short_udf = udf(is_short, BooleanType())
    df_flagged = df_filtered.withColumn("short_review", is_short_udf(col("review_clean")))
    short_count = df_flagged.filter(col("short_review")).count()

    # --- Write cleaned data to Spark-style output ---
    df_flagged.coalesce(1).write.mode("overwrite").option("header", True).csv(output_dir)

    # --- Flatten part-00000 to processed.csv ---
    response = s3.list_objects_v2(Bucket=bucket, Prefix=processed_dir_key + "/")
    part_file = next((obj["Key"] for obj in response.get("Contents", []) if "part-" in obj["Key"]), None)

    if part_file:
        s3.copy_object(
            Bucket=bucket,
            CopySource={'Bucket': bucket, 'Key': part_file},
            Key=processed_file_key
        )
        s3.delete_objects(Bucket=bucket, Delete={'Objects': [{'Key': obj["Key"]} for obj in response["Contents"]]})

    # --- Write summary JSON ---
    summary = {
        "original_rows": original_count,
        "dropped_null_or_empty": null_dropped,
        "dropped_numeric_only": numeric_dropped,
        "short_reviews_flagged": short_count,
        "retained_clean_reviews": df_flagged.count()
    }

    s3.put_object(Bucket=bucket, Key=summary_key, Body=json.dumps(summary, indent=2), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=status_key, Body=json.dumps({"status": "preprocessing_complete"}), ContentType="application/json")
    print("✅ Preprocessing complete.")

except Exception as e:
    print(f"❌ Preprocessing failed: {e}")
    mark_failed()
