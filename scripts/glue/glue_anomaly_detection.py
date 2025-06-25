import logging
import json
import boto3
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, length, when
from pyspark.sql.types import FloatType

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Input paths
input_path = "s3://mlops-sentiment-analysis-data/Bronze/schema_validated.parquet"
output_path = "s3://mlops-sentiment-analysis-data/Bronze/anomaly_flagged.parquet"
avg_length_metadata_path = "s3://mlops-sentiment-analysis-data/metadata/avg_review_length.json"

# Load data
df = spark.read.parquet(input_path)
logger.info(f"📥 Loaded records from: {input_path}")

# Load average review length from S3
s3 = boto3.client("s3")
bucket = "mlops-sentiment-analysis-data"
key = "metadata/avg_review_length.json"
avg_obj = s3.get_object(Bucket=bucket, Key=key)
avg_data = json.loads(avg_obj['Body'].read().decode('utf-8'))
avg_len = avg_data["avg_review_length"]
logger.info(f"📏 Loaded average review length: {avg_len:.2f}")

# Flag anomalies
df = df.withColumn("review_length", length(col("review_body")))
df = df.withColumn("anomaly_flag",
    when((col("review_length") < 10) | (col("review_length") > avg_len * 1.5), True).otherwise(False)
)

# Fix star_rating < 0 and > 5
df = df.withColumn("star_rating", col("star_rating").cast(FloatType()))
df = df.withColumn("star_rating",
    when(col("star_rating") < 0, None)  # log + null
    .when(col("star_rating") > 5, 5)    # scale to 5
    .otherwise(col("star_rating"))
)

invalid_ratings = df.filter(col("star_rating").isNull()).count()
logger.warning(f"⚠️ Set {invalid_ratings} ratings < 0 to null")
high_ratings = df.filter(col("star_rating") == 5).count()
logger.info(f"📏 Scaled down high ratings > 5 to 5 → total: {high_ratings}")

# Write result
df.write.mode("overwrite").parquet(output_path)
logger.info(f"✅ Anomaly detection complete. Output: {output_path}")
