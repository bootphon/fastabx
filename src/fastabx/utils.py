"""Various utilities."""

import json
import os
import queue
import threading
from collections.abc import Generator, Iterable

import torch

__all__ = ["InvalidEnvironmentVariableError"]


class InvalidEnvironmentVariableError(ValueError):
    """A ``FASTABX_*`` environment variable does not hold the kind of value it expects."""

    def __init__(self, name: str, value: str) -> None:
        super().__init__(
            f"The environment variable {name} must be a positive integer, but it is set to {value!r}. "
            f"Unset it to go back to the default."
        )


def positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to ``default`` when it is unset."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise InvalidEnvironmentVariableError(name, value) from None
    if parsed < 1:
        raise InvalidEnvironmentVariableError(name, value)
    return parsed


def max_score_chunk_rows() -> int:
    """Maximum number of rows compared at once when scoring a group of cells."""
    return positive_int_from_env("FASTABX_MAX_SCORE_CHUNK_ROWS", 8192)


def gather_chunk_rows() -> int:
    """Maximum number of rows gathered and padded in a single batched read from the accessor."""
    return positive_int_from_env("FASTABX_GATHER_CHUNK_ROWS", 8192)


def reduction_flush_cols() -> int:
    """Accumulated columns after which the per-cell reduction is flushed."""
    return positive_int_from_env("FASTABX_REDUCTION_FLUSH_COLS", 262144)


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve where the features are stored, from what the user asked for."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def with_librilight_bug() -> bool:
    """Whether to reproduce the results from LibriLight ABX or not."""
    return os.getenv("FASTABX_WITH_LIBRILIGHT_BUG", "0") == "1"


def hide_progress(*, progress: bool) -> bool:
    """Whether a progress bar must be hidden. ``TQDM_DISABLE`` takes precedence."""
    return bool(os.getenv("TQDM_DISABLE")) or not progress


def display_name(value: object) -> str:
    """Short readable name of a distance or an alignment, for ``repr`` and error messages."""
    if isinstance(value, str):
        return value
    return getattr(value, "__name__", type(value).__name__)


def print_fastabx_output(score: float, output: str = "text", **kwargs: str | int | None) -> None:
    """Help function to format fastabx CLI output."""
    match output:
        case "json":
            formatted = json.dumps(kwargs | {"score": score})
        case _:
            formatted = f"ABX error rate: {score:.3%}"
    print(formatted)


def prefetch[T](iterable: Iterable[T], maxsize: int = 1) -> Generator[T, None, None]:
    """Wrap an iterable, producing items ahead of consumption in a background thread.

    The producer thread is always cleaned up. On normal completion it ends on its own; if the
    consumer stops early (``break``, an exception, or the generator being closed), the ``finally``
    sets ``stop`` and drains the queue until the producer's guaranteed final sentinel, so no thread
    is left parked on a full ``put``.
    """
    q = queue.Queue(maxsize=maxsize)
    sentinel = object()
    stop = threading.Event()

    def producer() -> None:
        try:
            for item in iterable:
                if stop.is_set():
                    break
                q.put(item)
        except Exception as e:  # ruff: ignore[blind-except]
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
