import io
import boto3
from PyPDF2 import PdfReader
from app.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def fetch_pdf_files():
    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    return [f["Key"] for f in response.get("Contents", []) if f["Key"].endswith(".pdf")]

def extract_text_from_pdf(s3_key: str) -> str:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    reader = PdfReader(io.BytesIO(obj["Body"].read()))
    return "\n".join([page.extract_text() or "" for page in reader.pages])