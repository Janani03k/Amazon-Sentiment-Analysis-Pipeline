import boto3
import os
import threading

BUCKET_NAME = "mlops-sentiment-analysis-data"
FILE_PATH = "data/raw/reviews.csv"
KEY = "raw/reviews.csv"

s3_client = boto3.client("s3")
PART_SIZE = 50 * 1024 * 1024  # 50MB per part

def upload_part(upload_id, part_number, data):
    response = s3_client.upload_part(
        Bucket=BUCKET_NAME,
        Key=KEY,
        UploadId=upload_id,
        PartNumber=part_number,
        Body=data,
    )
    return {"PartNumber": part_number, "ETag": response["ETag"]}

def multi_part_upload():
    file_size = os.path.getsize(FILE_PATH)
    parts = []
    upload_id = s3_client.create_multipart_upload(Bucket=BUCKET_NAME, Key=KEY)["UploadId"]

    with open(FILE_PATH, "rb") as f:
        part_number = 1
        threads = []
        while chunk := f.read(PART_SIZE):
            thread = threading.Thread(
                target=lambda p, d: parts.append(upload_part(upload_id, p, d)),
                args=(part_number, chunk),
            )
            thread.start()
            threads.append(thread)
            part_number += 1

        for thread in threads:
            thread.join()

    s3_client.complete_multipart_upload(
        Bucket=BUCKET_NAME,
        Key=KEY,
        UploadId=upload_id,
        MultipartUpload={"Parts": sorted(parts, key=lambda x: x["PartNumber"])},
    )
    print(f"✅ Uploaded to s3://{BUCKET_NAME}/{KEY}")

if __name__ == "__main__":
    multi_part_upload()
