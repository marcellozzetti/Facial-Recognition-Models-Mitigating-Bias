"""CLI: detect, align and resize all images in the configured input directory."""

import argparse
import logging

from facenet_pytorch import MTCNN

from face_bias.config import load_config
from face_bias.preprocessing.pipeline import process_preprocessing
from face_bias.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the face detection and alignment pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config, "log_preprocessing_file")

    logging.info(f"Starting preprocessing with config: {args.config}")

    face_detector = MTCNN()
    process_preprocessing(
        config["data"]["dataset_image_input_path"],
        config["data"]["dataset_image_output_path"],
        face_detector,
        config,
    )

    logging.info("Preprocessing finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
