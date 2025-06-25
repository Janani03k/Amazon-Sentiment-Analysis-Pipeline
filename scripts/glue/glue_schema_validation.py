import logging
import json
import boto3
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, length, avg
from pyspark.sql.types import IntegerType

# Setup CloudWatch logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Glue + Spark Contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Paths
input_path = "s3://mlops-sentiment-analysis-data/Bronze/pre_processed.parquet"
output_path = "s3://mlops-sentiment-analysis-data/Bronze/schema_validated.parquet"
avg_length_metadata_path = "s3://mlops-sentiment-analysis-data/metadata/avg_review_length.json"

# Load preprocessed data
df = spark.read.parquet(input_path)
initial_count = df.count()
logger.info(f"📥 Loaded {initial_count} records from: {input_path}")

# Filter rows with non-null review_body and review_headline
df = df.filter((col("review_body").isNotNull()) & (col("review_headline").isNotNull()))
text_valid_count = df.count()
logger.info(f"✅ Records with non-null review_body and review_headline: {text_valid_count}")
logger.info(f"🗑️ Dropped {initial_count - text_valid_count} rows with missing text")

# Cast star_rating to int and log invalid ranges (don't change values)
df = df.withColumn("star_rating", col("star_rating").cast(IntegerType()))

# Log star_ratings < 0 and > 5
invalid_low = df.filter(col("star_rating") < 0).count()
invalid_high = df.filter(col("star_rating") > 5).count()
if invalid_low > 0:
    logger.warning(f"⚠️ Found {invalid_low} entries with star_rating < 0")
if invalid_high > 0:
    logger.warning(f"⚠️ Found {invalid_high} entries with star_rating > 5")

# Calculate and log average review_body length
df = df.withColumn("review_length", length(col("review_body")))
avg_len = df.select(avg("review_length")).first()[0]
logger.info(f"📏 Average review_body length: {avg_len:.2f}")

# Save average length to S3
avg_data = {"avg_review_length": avg_len}
s3 = boto3.client("s3")
s3.put_object(
    Bucket="mlops-sentiment-analysis-data",
    Key="metadata/avg_review_length.json",
    Body=json.dumps(avg_data),
    ContentType='application/json'
)
logger.info(f"📤 Stored average review length to: s3://mlops-sentiment-analysis-data/metadata/avg_review_length.json")

# Save schema validated data
df.write.mode("overwrite").parquet(output_path)
logger.info(f"✅ Schema validation complete. Output saved to: {output_path}")
