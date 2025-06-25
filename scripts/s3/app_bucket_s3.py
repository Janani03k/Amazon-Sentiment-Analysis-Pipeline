import boto3
import botocore
import traceback

s3 = boto3.client("s3")
s3_resource = boto3.resource("s3")

# ---------------------------
# Config
# ---------------------------
new_bucket = "mlops-sentiment-app"
source_bucket = "mlops-sentiment-analysis-data"
source_model_prefix = "models/"
dest_model_key = "models/model.tar.gz"

folders = [
    "uploads/raw/",
    "uploads/dropped/",
    "processed/cleaned/",
    "processed/tested/",
    "models/",
    "models/",
    "metadata/",
    "tmp/"
]

# ---------------------------
# Step 1: Create Bucket (if not exists)
# ---------------------------
def create_bucket(bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket already exists: {bucket_name}")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            s3.create_bucket(Bucket=bucket_name)
            print(f"✅ Created bucket: {bucket_name}")
        else:
            raise

# ---------------------------
# Step 2: Create Folder Prefixes
# ---------------------------
def prefix_exists(bucket_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
    return "Contents" in response

def create_folders(bucket_name, prefix_list):
    for prefix in prefix_list:
        if not prefix_exists(bucket_name, prefix):
            s3.put_object(Bucket=bucket_name, Key=prefix)
            print(f"📁 Created folder: {prefix}")
        else:
            print(f"📁 Folder already exists: {prefix}")

# ---------------------------
# Step 3: Find and Copy model.tar.gz from models/*/output/
# ---------------------------
def copy_latest_model():
    print(f"🔍 Searching in s3://{source_bucket}/{source_model_prefix}")
    bucket = s3_resource.Bucket(source_bucket)
    model_keys = []

    for obj in bucket.objects.filter(Prefix=source_model_prefix):
        if obj.key.endswith("output/model.tar.gz"):
            model_keys.append(obj.key)

    if not model_keys:
        print("❌ No model.tar.gz found under any models/*/output/")
        return

    print("📂 Found candidate models:")
    for k in model_keys:
        print(f"   - {k}")

    latest_key = sorted(model_keys)[-1]
    print(f"📦 Attempting to copy latest model: s3://{source_bucket}/{latest_key}")

    try:
        s3.copy_object(
            Bucket=new_bucket,
            CopySource={'Bucket': source_bucket, 'Key': latest_key},
            Key=dest_model_key
        )
        print(f"✅ Successfully copied to: s3://{new_bucket}/{dest_model_key}")
    except Exception as e:
        print("❌ Copy failed with exception:")
        traceback.print_exc()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    create_bucket(new_bucket)
    create_folders(new_bucket, folders)
    copy_latest_model()
