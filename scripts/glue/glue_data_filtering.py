from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, when
from pyspark.sql.types import FloatType
from pyspark.sql import DataFrame
import random

# Initialize Spark and Glue contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Input and output S3 paths
input_path = "s3://mlops-sentiment-analysis-data/raw/reviews.csv"
output_train = "s3://mlops-sentiment-analysis-data/Bronze/train.parquet"
output_test = "s3://mlops-sentiment-analysis-data/test/test.parquet"
output_serve = "s3://mlops-sentiment-analysis-data/serve/serve.parquet"

# Read the raw CSV from S3
df = spark.read.option("header", "true").csv(input_path)

# Select relevant columns
df = df.select(
    "product_category",
    "star_rating",
    "review_body",
    "review_headline"
)

# Remove records with null or empty review_body
df = df.filter((col("review_body").isNotNull()) & (col("review_body") != ""))

# Normalize star_rating to int in [1, 5]
df = df.withColumn(
    "star_rating",
    when(col("star_rating").isNotNull(), 
         when(col("star_rating").cast(FloatType()).isNotNull(), 
              when(col("star_rating").cast(FloatType()) < 1, 1)
              .when(col("star_rating").cast(FloatType()) > 5, 5)
              .otherwise(col("star_rating").cast(FloatType()).cast("int")))
         ).otherwise(None)
)

# Split into train (80%), test (10%), serve (10%)
train_df, test_df, serve_df = df.randomSplit([0.8, 0.1, 0.1], seed=42)

# Save each subset to respective S3 folders
train_df.write.mode("overwrite").parquet(output_train)
test_df.write.mode("overwrite").parquet(output_test)
serve_df.write.mode("overwrite").parquet(output_serve)

print("✅ Split and saved datasets to Bronze/train, test/, and serve/")
