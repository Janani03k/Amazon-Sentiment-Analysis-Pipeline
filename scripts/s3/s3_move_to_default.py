import boto3

s3 = boto3.client('s3')

bucket = 'mlops-sentiment-app'
source_key = 'uploads/raw/test.csv'
destination_key = 'uploads/raw/default/test.csv'

# Copy
s3.copy_object(
    Bucket=bucket,
    CopySource={'Bucket': bucket, 'Key': source_key},
    Key=destination_key
)

# Delete the original
s3.delete_object(Bucket=bucket, Key=source_key)

print("File moved successfully.")
