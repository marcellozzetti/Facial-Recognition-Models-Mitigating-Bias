"""CLI: download dataset archive from configured bucket."""

import argparse
import logging

from face_bias.config import load_config
from face_bias.data.bucket import download_from_bucket, load_bucket_client, upload_to_bucket
from face_bias.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download or upload the dataset archive.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config, "log_bucket_file")

    logging.info("Start of bucket processing")
    function = config["bucket"]["function"]

    if function == "download":
        logging.info("Download started")
        if not download_from_bucket(config):
            return 1
        logging.info("Download completed successfully")

    elif function == "upload":
        client = load_bucket_client(config)
        if client is None:
            return 1
        logging.info("Upload started")
        if not upload_to_bucket(config, client):
            return 1
        logging.info("Upload completed successfully")

    else:
        logging.error(f"Unknown bucket.function: {function!r} (expected 'download' or 'upload')")
        return 2

    logging.info("End of bucket processing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
