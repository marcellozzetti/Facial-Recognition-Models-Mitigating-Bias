"""CLI: print system + library + GPU report."""

import argparse
import logging

from face_bias.config import load_config
from face_bias.utils.logging import setup_logging
from face_bias.utils.system import system_info_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show system, library and GPU information.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_id = setup_logging(config, "log_version_file")

    system_info_report(["numpy", "pandas", "cv2", "torch", "torchvision", "sklearn"])
    logging.info(f"version report finished run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
