import io
import time
import boto3
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader
from typing import List
from app.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def fetch_pdf_files() -> List[str]:
    paginator = s3.get_paginator("list_objects_v2")
        # Use a concise list comprehension with pagination
    return [
        obj.get("Key")
        for page in paginator.paginate(Bucket=S3_BUCKET) # Iterate through all pages
        for obj in page.get("Contents", []) # Iterate through objects in the page (handles empty pages)
        if obj.get("Key") and obj.get("Key").endswith(".pdf") # Ensure Key exists and ends with .pdf
    ]

def extract_text_from_pdf(s3_key: str) -> str:
    retries = 3
    for attempt in range(retries):
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            reader = PdfReader(io.BytesIO(obj["Body"].read()))
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except ClientError as e:
            time.sleep(2 ** attempt)