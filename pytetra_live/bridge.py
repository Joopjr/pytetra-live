"""Optional direct bridge from validated bursts into PyTetra."""

import logging
import queue
import threading
import time

import numpy as np


LOG = logging.getLogger(__name__)


class PyTetraUnavailable(RuntimeError):
    pass


class PyTetraBridge:
    def __init__(self, debug=False):
        self.debug = bool(debug)
        self.stack = None
        self.resets = 0
        self._create_stack()

    def _create_stack(self):
        try:
            from pytetra.cli import ConsoleUserLayer
            from pytetra.logger import Logger
            from pytetra.stack import TetraStack
        except ImportError as exc:
            raise PyTetraUnavailable(
                "PyTetra is not importable. Install the sibling pytetra project "
                "or run PyTetra-live with --no-decode."
            ) from exc
        # Route every PyTetra line through this application's logging format.
        # This covers compact summaries, layer headings and debug diagnostics
        # and gives them the same live timestamp as receiver/DSP messages.
        Logger.set_writer(lambda message: LOG.info("%s", message))
        self.stack = TetraStack(ConsoleUserLayer, debug=self.debug)

    def reset(self):
        self.resets += 1
        reset_after_gap = getattr(self.stack, "reset_after_gap", None)
        if reset_after_gap is not None:
            reset_after_gap()
        else:
            self._create_stack()
        LOG.debug("PyTetra stream state reset after a demodulator gap")

    def feed_burst(self, burst, confidence=None):
        bits = [int(bit) for bit in burst]
        if confidence is not None and hasattr(self.stack.phy, "feed_soft"):
            self.stack.phy.feed_soft(bits, [float(value) for value in confidence])
        else:
            self.stack.phy.feed(bits)


class QueuedPyTetraBridge:
    """Keep ordered PyTetra decoding off the real-time IQ/DSP path."""

    def __init__(self, debug=False, capacity=512):
        self.bridge = PyTetraBridge(debug=debug)
        self.events = queue.Queue(maxsize=max(4, int(capacity)))
        self.overruns = 0
        self.error = None
        self._last_warning = 0.0
        self.worker = threading.Thread(
            target=self._run, name="pytetra-decoder", daemon=True
        )
        self.worker.start()

    @property
    def queue_depth(self):
        return self.events.qsize()

    def _run(self):
        try:
            while True:
                kind, payload = self.events.get()
                if kind == "stop":
                    return
                if kind == "reset":
                    self.bridge.reset()
                elif kind == "burst":
                    burst, confidence = payload
                    self.bridge.feed_burst(burst, confidence)
        except BaseException as exc:
            self.error = exc

    def _discard_pending(self):
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                return

    def _put(self, event):
        if self.error is not None:
            raise RuntimeError("PyTetra decoder worker failed") from self.error
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.overruns += 1
            self._discard_pending()
            self.events.put_nowait(("reset", None))
            if event[0] != "reset":
                self.events.put_nowait(event)
            now = time.monotonic()
            if now - self._last_warning >= 1.0:
                LOG.warning(
                    "PyTetra decoder queue overrun: stale bursts discarded"
                )
                self._last_warning = now

    def reset(self):
        self._put(("reset", None))

    def feed_burst(self, burst, confidence=None):
        hard = np.asarray(burst, dtype=np.uint8).copy()
        soft = (
            np.asarray(confidence, dtype=np.float32).copy()
            if confidence is not None
            else None
        )
        self._put(("burst", (hard, soft)))

    def close(self):
        if self.worker.is_alive():
            self.events.put(("stop", None))
            self.worker.join(timeout=10.0)
        if self.error is not None:
            raise RuntimeError("PyTetra decoder worker failed") from self.error


class BitFileSink:
    def __init__(self, path=None):
        self.path = path
        self.file = open(path, "ab", buffering=0) if path else None
        self.bits_written = 0

    def write(self, burst):
        if self.file is not None:
            self.file.write(bytes(int(bit) for bit in burst))
        self.bits_written += len(burst)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class IqFileSink:
    """Write normalized interleaved float32 IQ for reproducible diagnostics."""

    def __init__(self, path=None):
        self.path = path
        self.file = open(path, "ab", buffering=0) if path else None
        self.samples_written = 0

    def write(self, iq):
        if self.file is not None:
            interleaved = np.empty(len(iq) * 2, dtype="<f4")
            interleaved[0::2] = iq.real
            interleaved[1::2] = iq.imag
            self.file.write(interleaved.tobytes())
        self.samples_written += len(iq)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
