"""Various utilities."""

import json
import os
import queue
import threading
from collections.abc import Generator, Iterable

import torch

__all__ = []

# Environment variables to control fastabx behavior. Normal usage should not require changing these.
MIN_CELLS_FOR_TQDM = int(os.getenv("FASTABX_MIN_CELLS_FOR_TQDM", "50"))
MAX_SCORE_CHUNK_ROWS = int(os.getenv("FASTABX_MAX_SCORE_CHUNK_ROWS", "8192"))
GATHER_CHUNK_ROWS = int(os.getenv("FASTABX_GATHER_CHUNK_ROWS", "8192"))
REDUCTION_FLUSH_COLS = int(os.getenv("FASTABX_REDUCTION_FLUSH_COLS", "262144"))


def default_device() -> torch.device:
    """Return the default device used by fastabx: CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def prefetch[T](iterable: Iterable[T], maxsize: int = 1) -> Generator[T, None, None]:
    """Wrap an iterable, producing items ahead of consumption in a background thread.

    The producer thread is always cleaned up. On normal completion it ends on its own; if the
    consumer stops early (``break``, an exception, or the generator being closed), the ``finally``
    sets ``stop`` and drains the queue until the producer's guaranteed final sentinel, so no thread
    is left parked on a full ``put``.
    """
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    sentinel = object()
    stop = threading.Event()

    def producer() -> None:
        try:
            for item in iterable:
                if stop.is_set():
                    break
                q.put(item)
        except Exception as e:  # noqa: BLE001
            q.put(e)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    consumed_sentinel = False
    try:
        while (item := q.get()) is not sentinel:
            if isinstance(item, Exception):
                raise item
            yield item
        consumed_sentinel = True
    finally:
        stop.set()
        while not consumed_sentinel and q.get() is not sentinel:
            pass  # drain so a producer blocked on a full put can finish and exit
        thread.join()
