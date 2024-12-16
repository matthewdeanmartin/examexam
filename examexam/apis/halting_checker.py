import logging
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class FailureToHaltError(Exception):
    """Exception raised when a method is called more than a specified number of times."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def call_limit(limit: int) -> Callable:
    """
    A decorator factory that limits the number of times a function can be called.

    Args:
        limit (int): The maximum number of times the decorated function can be called.

    Returns:
        Callable: A decorator that enforces the call limit.
    """

    def decorator(func: Callable) -> Callable:
        # Initialize call count for the function
        func.call_count = 0  # type: ignore

        def wrapper(*args, **kwargs) -> Callable:
            func.call_count += 1  # type: ignore
            try:
                name = func.__name__
            except AttributeError:
                name = "unknown function"
            LOGGER.info(f"{name} called {func.call_count} times")  # type: ignore

            if func.call_count > limit:  # type: ignore
                raise FailureToHaltError(f"{name} has been called more than {limit} times")

            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    # Example usage
    @call_limit(3)
    def my_function():
        print("Function executed")

    for i in range(5):  # Attempt to call it 5 times.
        try:
            my_function()
            print(i)
        except FailureToHaltError as e:
            print(e)
            break
