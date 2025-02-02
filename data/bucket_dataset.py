import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import boto3
import zipfile
import requests
import io
import json
from botocore.exceptions import NoCredentialsError, ClientError
from utils.config import load_config
from utils.custom_logging import setup_logging

def load_config_bucket(config):
    """
    Load configuration from a YAML file related to Bucket
    """
    bucket_path_file = config['bucket']["bucket_path_file"]
    bucket_client = config['bucket']['bucket_client']

    os.makedirs(bucket_path_file, exist_ok=True)

    # Retrieve Bucket credentials from environment variables
    with open('configs/credentials.json') as cred_file:
        credentials = json.load(cred_file)

    bucket_access_key_id = credentials[bucket_client]['bucket_access_key_id']
    bucket_secret_access_key = credentials[bucket_client]['bucket_secret_access_key']

    if not bucket_access_key_id or not bucket_secret_access_key:
        logging.error("Bucket credentials are not set in environment variables.")
        return False

    # Initialize Bucket client
    client = boto3.client(
        bucket_client,
        aws_access_key_id=bucket_access_key_id,
        aws_secret_access_key=bucket_secret_access_key
    )

    return client

def upload_to_bucket(config, client):

    bucket_path_file = config['bucket']['bucket_path_file']
    bucket_name = config['bucket']['bucket_name']
    object_file_name = config['bucket']['object_file_name']

    file_path = os.path.join(bucket_path_file, object_file_name)

    try:
        client.upload_file(file_path, bucket_name, file_path)
        logging.info(f"File {bucket_path_file} successfully uploaded to {bucket_name}/{object_file_name}")
        return True
    except FileNotFoundError:
        logging.error(f"The file {bucket_path_file} was not found")
    except NoCredentialsError:
        logging.error("Bucket credentials are not available")
    except ClientError as e:
        logging.error(f"Client error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return False

def download_from_bucket(config, client):
    """
    Download a ZIP file from the given URL and extract it to the specified directory.
    """
    url = config['bucket']['bucket_download_url']
    bucket_path_file = config['bucket']['bucket_path_file']

    logging.info("Downloading dataset...")
    response = requests.get(url)

    if response.status_code == 200:
        logging.info(f"Dataset downloaded successfully")
        # Extract the ZIP content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(bucket_path_file)
        logging.info(f"Files extracted to {bucket_path_file}")
    else:
        logging.error(f"Error {response.status_code}: Failed to download the dataset.")

def main():
    # Load configuration
    config = load_config('configs/default.yaml')
    setup_logging(config, 'log_bucket_file')

    logging.info("Start of bucket processing")
    client = load_config_bucket(config)

    if(config['bucket']['function'] == 'upload'):
        logging.info("Upload started")
        upload_to_bucket(config, client)
        logging.info("Upload completed successfully")

    if(config['bucket']['function'] == 'download'):
        logging.info("Download started")
        download_from_bucket(config, client)
        logging.info("Download completed successfully")

    logging.info("End of bucket processing")

if __name__ == "__main__":
    main()