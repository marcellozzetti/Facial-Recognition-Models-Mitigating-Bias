import os
import boto3
import logging
import yaml
from botocore.exceptions import NoCredentialsError, ClientError

def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def upload_to_s3(file_path, config, bucket, object_name=None):
    """
    Upload a file to an S3 bucket.
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    # Retrieve AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not aws_access_key_id or not aws_secret_access_key:
        logging.error("AWS credentials are not set in environment variables.")
        return False

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config['aws']['access_key'],
        aws_secret_access_key=config['aws']['secret_access_key']
    )

    try:
        s3_client.upload_file(file_path, bucket, object_name)
        logging.info(f"File {file_path} successfully uploaded to {bucket}/{object_name}")
        return True
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
    except NoCredentialsError:
        logging.error("AWS credentials are not available.")
    except ClientError as e:
        logging.error(f"Client error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return False

def main():
    # Load configuration
    config = load_config('configs/default.yaml')

    # Set up logging
    logging.basicConfig(
        level=config['logging']['log_level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['log_file']),
            logging.StreamHandler()
        ]
    )

    # File and S3 parameters from config
    file_path = config['data']['file_path']
    file_name = config['data']['file_name']
    bucket_name = config['aws']['bucket_name']
    object_name = config['aws'].get('object_name', file_name)

    # Upload file to S3
    success = upload_to_s3(os.path.join(file_path, file_name), config, bucket_name, object_name)
    if success:
        logging.info("Upload completed successfully.")
    else:
        logging.error("Upload failed.")

if __name__ == "__main__":
    main()