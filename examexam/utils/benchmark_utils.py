import io
import logging
import sys
import time
from functools import wraps
from typing import Any


class CaptureOutput:
    """Context manager that captures the standard output of a function."""

    def __init__(self) -> None:
        self.old_output = None
        self.captured_output = None

    def __enter__(self) -> io.StringIO:
        """Enter the runtime context related to this object.

        Returns:
            StringIO: The captured output.
        """
        self.captured_output = io.StringIO()
        self.old_output = sys.stdout
        sys.stdout = self.captured_output
        return self.captured_output

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context and return a Boolean flagging an exception.

        Args:
            exc_type: The type of the exception.
            exc_val: The value of the exception.
            exc_tb: The traceback of the exception.
        """
        if self.old_output:
            sys.stdout = self.old_output
        if self.captured_output:
            self.captured_output.close()


def log_duration(func: Any) -> Any:
    """
    A decorator that logs the duration of a method's execution in a human-readable format.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds

        if duration < 500:
            logging.info(f"{func.__name__} execution time: {duration:.2f} milliseconds")
        elif duration < 60000:
            logging.info(f"{func.__name__} execution time: {duration / 1000:.2f} seconds")
        else:
            logging.info(f"{func.__name__} execution time: {duration / 60000:.2f} minutes")
        return result

    return wrapper
