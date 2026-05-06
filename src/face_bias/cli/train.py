"""CLI placeholder for the training pipeline (Sprint C)."""

import argparse


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    parser = argparse.ArgumentParser(description="Train a face recognition model (Sprint C).")
    parser.parse_args(argv)
    raise NotImplementedError(
        "Training pipeline will be implemented in Sprint C. See REVIEW_AND_PLAN.md §5."
    )


if __name__ == "__main__":
    raise SystemExit(main())
