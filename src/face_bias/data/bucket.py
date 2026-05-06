"""Object-storage client (S3 / Azure Blob) for dataset download/upload.

Credentials are loaded from configs/credentials.json (gitignored) or from
environment variables — never from configs/default.yaml.
"""

import io
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError

CREDENTIALS_PATH = Path("configs/credentials.json")


def _load_credentials(bucket_client: str) -> dict[str, str]:
    """Load credentials from JSON file or env vars (env vars take precedence)."""
    creds: dict[str, str] = {}

    if CREDENTIALS_PATH.exists():
        with open(CREDENTIALS_PATH) as cred_file:
            data = json.load(cred_file)
            creds = data.get(bucket_client, {})

    env_overrides = {
        "bucket_access_key_id": os.environ.get("FACE_BIAS_BUCKET_ACCESS_KEY_ID"),
        "bucket_secret_access_key": os.environ.get("FACE_BIAS_BUCKET_SECRET_ACCESS_KEY"),
        "bucket_session_token": os.environ.get("FACE_BIAS_BUCKET_SESSION_TOKEN"),
        "bucket_download_url": os.environ.get("FACE_BIAS_BUCKET_URL"),
    }
    for key, value in env_overrides.items():
        if value:
            creds[key] = value

    return creds


def load_bucket_client(config: dict[str, Any]) -> Any:
    """Initialise a boto3 client for the configured bucket."""
    bucket_path_file = config["bucket"]["bucket_path_file"]
    bucket_client_name = config["bucket"]["bucket_client"]

    os.makedirs(bucket_path_file, exist_ok=True)

    creds = _load_credentials(bucket_client_name)
    access_key = creds.get("bucket_access_key_id")
    secret_key = creds.get("bucket_secret_access_key")
    session_token = creds.get("bucket_session_token")

    if not access_key or not secret_key:
        logging.error(
            "Bucket credentials not found. Provide configs/credentials.json or env vars "
            "FACE_BIAS_BUCKET_ACCESS_KEY_ID / FACE_BIAS_BUCKET_SECRET_ACCESS_KEY."
        )
        return None

    return boto3.client(
        bucket_client_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
    )


def upload_to_bucket(config: dict[str, Any], client: Any) -> bool:
    """Upload a local archive to the configured bucket."""
    bucket_path_file = config["bucket"]["bucket_path_file"]
    bucket_name = config["bucket"]["bucket_name"]
    object_file_name = config["bucket"]["object_file_name"]

    file_path = os.path.join(bucket_path_file, object_file_name)

    try:
        client.upload_file(file_path, bucket_name, object_file_name)
        logging.info(f"File {file_path} uploaded to {bucket_name}/{object_file_name}")
        return True
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found")
    except NoCredentialsError:
        logging.error("Bucket credentials are not available")
    except ClientError as exc:
        logging.error(f"Client error: {exc}")
    except Exception as exc:  # noqa: BLE001 — defensive at integration boundary
        logging.error(f"An unexpected error occurred: {exc}")

    return False


def download_from_bucket(config: dict[str, Any]) -> bool:
    """Download a ZIP archive via presigned URL and extract it locally.

    The presigned URL must come from configs/credentials.json or the
    FACE_BIAS_BUCKET_URL environment variable.
    """
    bucket_client_name = config["bucket"]["bucket_client"]
    bucket_path_file = config["bucket"]["bucket_path_file"]

    creds = _load_credentials(bucket_client_name)
    url = creds.get("bucket_download_url")
    if not url:
        logging.error(
            "No bucket_download_url found. Set it in configs/credentials.json "
            "or via FACE_BIAS_BUCKET_URL env var."
        )
        return False

    logging.info("Downloading dataset...")
    response = requests.get(url, timeout=300)

    if response.status_code != 200:
        logging.error(f"Error {response.status_code}: failed to download the dataset.")
        return False

    logging.info("Dataset downloaded; extracting...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(bucket_path_file)
    logging.info(f"Files extracted to {bucket_path_file}")
    return True
