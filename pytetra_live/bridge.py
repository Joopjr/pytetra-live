"""Optional direct bridge from validated bursts into PyTetra."""

import logging
import multiprocessing
import queue
import signal
import threading
import time
import traceback

import numpy as np


LOG = logging.getLogger(__name__)
PYTETRA_LEVEL = 25
logging.addLevelName(PYTETRA_LEVEL, "PYTETRA")


def log_pytetra_output(message):
    """Give decoded Layer-2/3 records their own log category."""
    message = str(message)
    compact = message.startswith("DL;") and (
        "; Layer 2 -" in message or "; Layer 3 -" in message
    )
    detailed = (
        "Layer 2 - MAC / LLC" in message
        or "Layer 3 - MLE / CMCE / MM / SNDCP" in message
        or message.startswith((
            "LowerMac:",
            "UpperMac:",
            "Llc:",
            "Mle:",
            "Cmce:",
            "Mm:",
            "Sndcp:",
        ))
    )
    if compact or detailed:
        LOG.log(PYTETRA_LEVEL, "%s", message)
    else:
        LOG.info("%s", message)


class PyTetraUnavailable(RuntimeError):
    pass


class PyTetraBridge:
    def __init__(
        self,
        debug=False,
        line_writer=None,
        show_esi=False,
        show_security_context=False,
    ):
        self.debug = bool(debug)
        self.line_writer = line_writer
        self.show_esi = bool(show_esi)
        self.show_security_context = bool(show_security_context)
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
        Logger.set_writer(
            self.line_writer or log_pytetra_output
        )
        self.stack = TetraStack(
            ConsoleUserLayer,
            debug=self.debug,
            show_esi=self.show_esi,
            show_security_context=self.show_security_context,
        )

    def reset(self):
        self.resets += 1
        reset_after_gap = getattr(self.stack, "reset_after_gap", None)
        if reset_after_gap is not None:
            reset_after_gap()
        else:
            self._create_stack()
        LOG.debug("PyTetra stream state reset after a demodulator gap")

    def feed_burst(self, burst, confidence=None):
        if confidence is not None and hasattr(self.stack.phy, "feed_soft"):
            # PyTetra performs its own validated conversion. Avoid duplicating
            # 1020 Python scalar conversions for every live burst.
            self.stack.phy.feed_soft(burst, confidence)
        else:
            self.stack.phy.feed(burst)


def _event_burst_count(event):
    kind, payload = event
    if kind == "burst":
        return 1
    if kind == "burst_batch":
        return int(payload[2])
    return 0


def _change_counter(counter, delta):
    with counter.get_lock():
        counter.value = max(0, counter.value + int(delta))


def _decoder_process(
    events,
    output,
    errors,
    queued_bursts,
    debug,
    show_esi,
    show_security_context,
):
    """Run stateful PyTetra decoding outside the DSP interpreter process."""
    try:
        # The parent owns terminal signals and requests an ordered shutdown.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        bridge = PyTetraBridge(
            debug=debug,
            line_writer=output.put,
            show_esi=show_esi,
            show_security_context=show_security_context,
        )
        while True:
            event = events.get()
            _change_counter(queued_bursts, -_event_burst_count(event))
            kind, payload = event
            if kind == "stop":
                return
            if kind == "reset":
                bridge.reset()
            elif kind == "burst":
                hard_bytes, soft_bytes = payload
                hard = np.frombuffer(hard_bytes, dtype=np.uint8)
                soft = (
                    np.frombuffer(soft_bytes, dtype=np.float32)
                    if soft_bytes is not None
                    else None
                )
                bridge.feed_burst(hard, soft)
            elif kind == "burst_batch":
                hard_bytes, soft_bytes, count = payload
                hard = np.frombuffer(hard_bytes, dtype=np.uint8).reshape(
                    int(count), -1
                )
                soft = (
                    np.frombuffer(soft_bytes, dtype=np.float32).reshape(
                        int(count), -1
                    )
                    if soft_bytes is not None
                    else None
                )
                for index in range(int(count)):
                    bridge.feed_burst(
                        hard[index],
                        soft[index] if soft is not None else None,
                    )
    except BaseException:
        errors.put(traceback.format_exc())
    finally:
        output.put(None)


class QueuedPyTetraBridge:
    """Decode PyTetra in a separate process without blocking real-time DSP."""

    def __init__(
        self,
        debug=False,
        capacity=512,
        show_esi=False,
        show_security_context=False,
        _worker_target=None,
    ):
        context = multiprocessing.get_context("spawn")
        self.events = context.Queue(maxsize=max(4, int(capacity)))
        # Output is intentionally unbounded: terminal I/O must never stall the
        # protocol decoder or, indirectly, the DSP pipeline.
        self.output = context.Queue()
        self.errors = context.Queue(maxsize=1)
        self.overruns = 0
        self.dropped_bursts = 0
        self.queued_bursts = context.Value("q", 0)
        self._last_warning = 0.0
        self._reset_pending = False
        self.worker = context.Process(
            target=_worker_target or _decoder_process,
            args=(
                self.events,
                self.output,
                self.errors,
                self.queued_bursts,
                bool(debug),
                bool(show_esi),
                bool(show_security_context),
            ),
            name="pytetra-decoder",
            daemon=True,
        )
        self.worker.start()
        self.output_worker = threading.Thread(
            target=self._forward_output,
            name="pytetra-log-forwarder",
            daemon=True,
        )
        self.output_worker.start()

    @property
    def queue_depth(self):
        try:
            return self.events.qsize()
        except (NotImplementedError, OSError):
            return 0

    @property
    def queue_bursts(self):
        with self.queued_bursts.get_lock():
            return int(self.queued_bursts.value)

    def _forward_output(self):
        while True:
            message = self.output.get()
            if message is None:
                return
            log_pytetra_output(message)

    def _raise_worker_error(self):
        try:
            error = self.errors.get_nowait()
        except queue.Empty:
            error = None
        if error is not None:
            raise RuntimeError("PyTetra decoder process failed:\n%s" % error)
        if (
            not self.worker.is_alive()
            and self.worker.exitcode is not None
            and self.worker.exitcode != 0
        ):
            raise RuntimeError(
                "PyTetra decoder process stopped with exit code %d"
                % self.worker.exitcode
            )

    def _discard_pending(self):
        discarded = 0
        while True:
            try:
                discarded += _event_burst_count(self.events.get_nowait())
            except queue.Empty:
                _change_counter(self.queued_bursts, -discarded)
                return discarded

    def _put(self, event):
        self._raise_worker_error()
        if self._reset_pending:
            try:
                self.events.put_nowait(("reset", None))
            except queue.Full:
                self._record_overrun()
                return
            self._reset_pending = False
        burst_count = _event_burst_count(event)
        # Increment before publishing the event: the worker may consume it
        # immediately on another CPU.
        _change_counter(self.queued_bursts, burst_count)
        try:
            self.events.put_nowait(event)
        except queue.Full:
            _change_counter(self.queued_bursts, -burst_count)
            discarded = self._discard_pending()
            try:
                self.events.put_nowait(("reset", None))
            except queue.Full:
                # multiprocessing.Queue may report full before its feeder
                # makes discarded items visible. Defer reset and drop this
                # event; overload must never terminate the IQ receiver.
                self._reset_pending = True
                self._record_overrun(discarded + burst_count)
                return
            if event[0] != "reset":
                _change_counter(self.queued_bursts, burst_count)
                try:
                    self.events.put_nowait(event)
                except queue.Full:
                    _change_counter(self.queued_bursts, -burst_count)
                    # The reset is already ordered. Dropping this burst is
                    # safe and prevents a second queue.Full from escaping.
                    self._record_overrun(discarded + burst_count)
                    return
            self._record_overrun(discarded)

    def _record_overrun(self, dropped_bursts=0):
        self.overruns += 1
        self.dropped_bursts += int(dropped_bursts)
        now = time.monotonic()
        if now - self._last_warning >= 1.0:
            LOG.warning(
                "PyTetra decoder queue overrun: %d bursts discarded in total",
                self.dropped_bursts,
            )
            self._last_warning = now

    def reset(self):
        self._put(("reset", None))

    def feed_burst(self, burst, confidence=None):
        # Immutable bytes are serialized safely even if the caller reuses its
        # NumPy buffers before multiprocessing's feeder thread runs.
        hard = np.asarray(burst, dtype=np.uint8).tobytes()
        soft = (
            np.asarray(confidence, dtype=np.float32).tobytes()
            if confidence is not None
            else None
        )
        self._put(("burst", (hard, soft)))

    def feed_bursts(self, bursts, confidences=None):
        """Send an ordered DSP batch through one multiprocessing event."""
        count = len(bursts)
        if not count:
            return
        hard_values = np.asarray(bursts, dtype=np.uint8)
        if hard_values.ndim != 2:
            raise ValueError("Burst batch must be a two-dimensional array")
        soft_values = None
        if confidences is not None:
            soft_values = np.asarray(confidences, dtype=np.float32)
            if soft_values.shape != hard_values.shape:
                raise ValueError("Confidence batch must match burst batch")
        self._put((
            "burst_batch",
            (
                hard_values.tobytes(),
                soft_values.tobytes() if soft_values is not None else None,
                count,
            ),
        ))

    def close(self):
        if self.worker.is_alive():
            # FIFO ordering guarantees that every accepted burst is decoded
            # before stop. The optimized PyTetra worker drains this bounded
            # queue without Ctrl-C reaching the child mid-Viterbi operation.
            self.events.put(("stop", None))
            self.worker.join(timeout=30.0)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=2.0)
            raise RuntimeError("PyTetra decoder process did not stop cleanly")
        self.output_worker.join(timeout=2.0)
        self._raise_worker_error()


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
