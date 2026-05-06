"""CLI placeholder for the evaluation pipeline (Sprint C)."""

import argparse


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    parser = argparse.ArgumentParser(description="Evaluate a trained model (Sprint C).")
    parser.parse_args(argv)
    raise NotImplementedError(
        "Evaluation pipeline will be implemented in Sprint C. See REVIEW_AND_PLAN.md §5."
    )


if __name__ == "__main__":
    raise SystemExit(main())
