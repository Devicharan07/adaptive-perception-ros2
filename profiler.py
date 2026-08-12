import time

class Profiler:
    def __init__(self):
        self._start_time = None
        self._stage_start_times = {}
        self._stage_times_ms = {}

    def start(self):
        self._start_time = time.perf_counter()
        self._stage_start_times = {}
        self._stage_times_ms = {}

    def stop(self) -> float:
        if self._start_time is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._start_time = None
        return elapsed_ms

    def start_stage(self, name: str):
        self._stage_start_times[name] = time.perf_counter()

    def stop_stage(self, name: str) -> float:
        start_time = self._stage_start_times.pop(name, None)
        if start_time is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._stage_times_ms[name] = elapsed_ms
        return elapsed_ms

    def stage_ms(self, name: str) -> float:
        return self._stage_times_ms.get(name, 0.0)
