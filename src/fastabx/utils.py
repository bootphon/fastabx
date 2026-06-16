"""Various utilities."""

import json
import os
import queue
import threading
from collections.abc import Iterable, Iterator

# Environment variables to control fastabx behavior. Normal usage should not require changing these.
MIN_CELLS_FOR_TQDM = int(os.getenv("FASTABX_MIN_CELLS_FOR_TQDM", "50"))
MAX_SCORE_CHUNK_ROWS = int(os.getenv("FASTABX_MAX_SCORE_CHUNK_ROWS", "8192"))
GATHER_CHUNK_ROWS = int(os.getenv("FASTABX_GATHER_CHUNK_ROWS", "8192"))
REDUCTION_FLUSH_COLS = int(os.getenv("FASTABX_REDUCTION_FLUSH_COLS", "262144"))


def with_librilight_bug() -> bool:
    """Whether to reproduce the results from LibriLight ABX or not."""
    return os.getenv("FASTABX_WITH_LIBRILIGHT_BUG", "0") == "1"


def print_fastabx_output(score: float, **kwargs: str | int) -> None:
    """Help function to format fastabx CLI output."""
    match os.getenv("FASTABX_OUTPUT"):
        case "json" | "jsonl":
            output = json.dumps(kwargs | {"score": score})
        case _:
            output = f"ABX error rate: {score:.3%}"
    print(output)  # noqa: T201


def prefetch[T](iterable: Iterable[T], maxsize: int = 1) -> Iterator[T]:
    """Wrap an iterator, producing items ahead of consumption in a background thread."""
    q = queue.Queue(maxsize=maxsize)
    sentinel = object()

    def producer() -> None:
        try:
            for item in iterable:
                q.put(item)
        except Exception as e:  # noqa: BLE001
            q.put(e)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    while (item := q.get()) is not sentinel:
        if isinstance(item, Exception):
            raise item from item
        yield item
    thread.join()
