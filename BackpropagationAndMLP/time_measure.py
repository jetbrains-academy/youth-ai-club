import time


def print_time(function, time, who):
    print(f"{who}'s max time on {function}: {time * 1000:.2f} ms")


class TimeMeasure:
    def __init__(self):
        self.start = None
        self.end = None

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.start = time.perf_counter()
            result = func(*args, **kwargs)
            self.end = time.perf_counter()
            return result

        return wrapper

    def get(self):
        return self.end - self.start
