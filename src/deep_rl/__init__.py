import time


def timed_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        if isinstance(result, tuple):
            return *result, elapsed_time
        else:
            return result, elapsed_time

    return wrapper
