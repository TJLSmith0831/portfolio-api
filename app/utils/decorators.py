import time
import logging
from functools import wraps

log = logging.getLogger("timing")


def timed(label: str | None = None, extra: dict | None = None):
    """
    Decorator to log execution time of a function.

    :param label: Optional label override. Defaults to function name.
    :param extra: Optional metadata dict to log.
    """
    def decorator(func):
        name = label or func.__qualname__

        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                ms = (time.perf_counter() - t0) * 1000
                log.info(
                    "TIMING | %s | %.1f ms | %s",
                    name,
                    ms,
                    extra or {},
                )

        return wrapper

    return decorator

if __name__ == "__main__":
    pass
