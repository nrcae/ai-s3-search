import io
import concurrent.futures
import boto3
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader
from typing import List, Dict, Any
from app.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def fetch_page(bucket: str, prefix: str = '', continuation_token: str | None = None) -> Dict[str, Any]:
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    if continuation_token:
        kwargs['ContinuationToken'] = continuation_token
    try:
        response = s3.list_objects_v2(**kwargs)
        return response
    except Exception as e:
        logger.error(f" Fetching S3 page (token: {continuation_token}): {e}")
        return {} # Return empty dict on error to avoid breaking the loop

def fetch_pdf_files(max_workers: int = 10) -> List[str]: # Added max_workers parameter
    logger.info(f" Fetching PDF file list from s3://{S3_BUCKET}/ concurrently (max_workers={max_workers})...")
    pdf_keys: List[str] = []
    tokens_to_fetch: List[str] = []
    s3_prefix = ''

    # 1. Fetch the first page to start and get initial keys/token
    try:
        response = fetch_page(S3_BUCKET, prefix=s3_prefix)
        page_keys = [
            obj.get("Key")
            for obj in response.get("Contents", [])
            if obj.get("Key") and obj.get("Key").endswith(".pdf")
        ]
        pdf_keys.extend(page_keys)
        is_truncated = response.get('IsTruncated', False)
        next_token = response.get('NextContinuationToken')
    except Exception as e:
        logger.error(f" Fetching initial S3 page: {e}")
        return []

    # 2. Sequentially collect all subsequent continuation tokens
    # This sequential step is necessary as each response contains the *next* token.
    while is_truncated and next_token:
        tokens_to_fetch.append(next_token)
        try:
            # We only need the token here, so a quick request is okay
            response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix, ContinuationToken=next_token)
            is_truncated = response.get('IsTruncated', False)
            next_token = response.get('NextContinuationToken')
        except Exception as e:
            logger.error(f" Fetching next continuation token ({next_token}): {e}")
            # Stop collecting tokens if an error occurs during pagination check
            break

    logger.debug(f" Found {len(tokens_to_fetch)} additional pages to fetch concurrently.")

    # 3. Fetch the remaining pages concurrently
    if tokens_to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to fetch each page using its token
            future_to_token = {
                executor.submit(fetch_page, S3_BUCKET, prefix=s3_prefix, continuation_token=token): token
                for token in tokens_to_fetch
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_token):
                token = future_to_token[future]
                try:
                    page = future.result()
                    page_keys = [
                        obj.get("Key")
                        for obj in page.get("Contents", [])
                        if obj.get("Key") and obj.get("Key").endswith(".pdf")
                    ]
                    pdf_keys.extend(page_keys)
                except Exception as exc:
                    logger.error(f" Fetching page with token {token} generated an exception: {exc}")

    logger.info(f" Found total {len(pdf_keys)} PDF files.")
    return pdf_keys
    
def extract_text_from_pdf(s3_key: str) -> str:
    try:
        # Body is a StreamingBody and avoid loading the entire PDF into memory.
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        pdf_content_bytes = response["Body"].read()

        # Wrap the in-memory bytes with io.BytesIO to make it seekable
        pdf_file_like_object = io.BytesIO(pdf_content_bytes)
        # Process the BytesIO object with PdfReader
        reader = PdfReader(pdf_file_like_object)

        # Extract text using a generator expression within join
        extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        return extracted_text

    # Catch both specific S3 errors and general processing errors
    except (ClientError, Exception) as e:
        # Log the error for debugging
        logger.error(f" Processing PDF '{s3_key}': {e}")
        # Return an empty string consistently on failure
        return ""