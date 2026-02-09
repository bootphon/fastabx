"""Various utilities."""

import json
import os

MIN_CELLS_FOR_TQDM = 50


def with_librilight_bug() -> bool:
    """Whether to reproduce the results from LibriLight ABX or not."""
    return os.getenv("FASTABX_WITH_LIBRILIGHT_BUG", "0") == "1"


def print_fastabx_output(score: float, **kwargs: str | int) -> None:
    """Help function to format fastabx CLI output."""
    match os.getenv("FASTABX_OUTPUT"):
        case "jsonl":
            output = json.dumps(kwargs | {"score": score})
        case _:
            output = f"ABX error rate: {score:.3%}"
    print(output)  # noqa: T201
