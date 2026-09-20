"""Bounded per-recording workers; capture never waits for disk writes."""

import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)
DEPTH_BUFFER_BYTES = 256 * 1024 * 1024
VIDEO_BUFFER_BYTES = 16 * 1024 * 1024


class RecordingBuffer:
    """Own submitted payloads until saved. Budget includes the in-flight item.

    Python/codec overhead is additional. Never overwrite accepted frames;
    overflow and write errors mark the recording incomplete.
    """

    def __init__(self, name, capacity_bytes, write, finish=None):
        if capacity_bytes <= 0:
            raise ValueError("Recording buffer capacity must be positive")
        self.name, self.capacity_bytes = name, capacity_bytes
        self._write, self._finish = write, finish
        self._pending = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._bytes = self._peak = self._rejected = self._errors = self._written = 0
        self._error = ""
        self._thread = threading.Thread(target=self._run, name=f"record-{name}", daemon=True)
        self._thread.start()

    def submit(self, payload, size):
        if size <= 0:
            raise ValueError("Recording payload size must be positive")
        with self._condition:
            if self._closed:
                raise RuntimeError("Recording buffer is closed")
            if self._bytes + size > self.capacity_bytes:
                self._rejected += 1
                if self._rejected == 1:
                    logger.error("%s recording buffer full; recording is incomplete", self.name)
                return False
            self._bytes += size
            self._peak = max(self._peak, self._bytes)
            self._pending.append((payload, size))
            self._condition.notify()
            return True

    def _failed(self, exc):
        with self._condition:
            self._errors += 1
            self._error = str(exc)
            if self._errors == 1:
                logger.exception("%s recording write failed; recording is incomplete", self.name)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    break
                payload, size = self._pending.popleft()
            try:
                self._write(payload)
                with self._condition:
                    self._written += 1
            except Exception as exc:
                self._failed(exc)
            finally:
                del payload
                with self._condition:
                    self._bytes -= size
        if self._finish is not None:
            try:
                self._finish()
            except Exception as exc:
                self._failed(exc)

    def close(self):
        """Seal, drain, and close output before the caller finalizes files."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()
        stats = self.stats()
        logger.info("%s buffer peak %.1f%% (%d/%d bytes), rejected=%d, write_errors=%d",
                    self.name, stats['peak_percent'], self._peak, self.capacity_bytes,
                    self._rejected, self._errors)
        return stats

    def stats(self):
        with self._condition:
            return {"capacity_bytes": self.capacity_bytes, "peak_bytes": self._peak,
                    "peak_percent": round(100 * self._peak / self.capacity_bytes, 2),
                    "pending_bytes": self._bytes, "written_frames": self._written,
                    "rejected_frames": self._rejected, "write_errors": self._errors,
                    "error": self._error,
                    "complete": not (self._rejected or self._errors)}
