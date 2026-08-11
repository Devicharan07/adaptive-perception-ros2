import time

class Profiler:
    def __init__(self):
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        if self._start_time is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._start_time = None
        return elapsed_ms
