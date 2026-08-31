"""Orchestrate SpyServer acquisition, demodulation, and PyTetra delivery."""

from dataclasses import dataclass
import logging
import queue
import threading
import time

import numpy as np

from .bridge import BitFileSink, IqFileSink, PyTetraBridge
from .client import SpyServerClient, SpyServerDisconnected
from .dsp import LiveTetraDemodulator


LOG = logging.getLogger(__name__)


@dataclass
class ReceiverStats:
    connections: int = 0
    reconnects: int = 0
    iq_samples: int = 0
    bursts: int = 0
    sequence_gaps: int = 0
    queue_overruns: int = 0


class LiveReceiver:
    # Keep enough elasticity for short scheduler stalls without accumulating
    # seconds of latency when the decoder cannot sustain real-time operation.
    IQ_QUEUE_BLOCKS = 64

    def __init__(
        self,
        host,
        port,
        frequency,
        center_frequency=None,
        gain=None,
        sample_rate=None,
        debug=False,
        decode=True,
        bits_output=None,
        iq_output=None,
        reconnect=True,
        reconnect_delay=2.0,
        timeout=5.0,
    ):
        self.host = host
        self.port = int(port)
        self.frequency = int(frequency)
        self.center_frequency = center_frequency
        self.gain = gain
        self.sample_rate = sample_rate
        self.debug = bool(debug)
        self.bridge = PyTetraBridge(debug=debug) if decode else None
        self.bits = BitFileSink(bits_output)
        self.iq = IqFileSink(iq_output)
        self.reconnect = bool(reconnect)
        self.reconnect_delay = float(reconnect_delay)
        self.timeout = float(timeout)
        self.stats = ReceiverStats()
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def _run_connection(self, duration=None):
        client = SpyServerClient(self.host, self.port, timeout=self.timeout)
        started = time.monotonic()
        try:
            client.connect()
            self.stats.connections += 1
            info = client.device_info
            sync = client.client_sync
            LOG.info(
                "SpyServer device: type=%d serial=%08X max_rate=%d Hz "
                "max_bandwidth=%d Hz decimation=%d..%d",
                info.device_type,
                info.device_serial,
                info.maximum_sample_rate,
                info.maximum_bandwidth,
                info.minimum_iq_decimation,
                info.decimation_stage_count,
            )
            LOG.info(
                "SpyServer state: device_center=%d Hz iq_center=%d Hz "
                "iq_range=%d..%d Hz gain=%d control=%s",
                sync.device_center_frequency,
                sync.iq_center_frequency,
                sync.minimum_iq_center_frequency,
                sync.maximum_iq_center_frequency,
                sync.gain,
                sync.can_control,
            )
            configuration = client.configure(
                self.frequency,
                center_frequency=self.center_frequency,
                gain=self.gain,
                sample_rate=self.sample_rate,
            )
            LOG.info(
                "SpyServer connected: rate=%.3f Hz iq_center=%d Hz channel=%d Hz "
                "local_offset=%+.0f Hz gain=%d control=%s",
                configuration.sample_rate,
                configuration.iq_center_frequency,
                configuration.frequency,
                configuration.frequency - configuration.iq_center_frequency,
                configuration.gain,
                configuration.can_control,
            )
            demodulator = LiveTetraDemodulator(
                configuration.sample_rate,
                channel_offset=(
                    configuration.frequency - configuration.iq_center_frequency
                ),
                nominal_frequency=configuration.frequency,
            )
            client.start()
            iq_queue = queue.Queue(maxsize=self.IQ_QUEUE_BLOCKS)
            reader_done = threading.Event()
            reader_error = []
            generation = [0]
            full_reset_pending = [False]
            generation_lock = threading.Lock()

            def discard_queued_iq():
                while True:
                    try:
                        iq_queue.get_nowait()
                    except queue.Empty:
                        return

            def receive_iq():
                observed_gaps = client.sequence_gaps
                try:
                    for block in client.iq_blocks():
                        if self.stop_requested:
                            break
                        if client.sequence_gaps != observed_gaps:
                            observed_gaps = client.sequence_gaps
                            with generation_lock:
                                generation[0] += 1
                                full_reset_pending[0] = True
                            discard_queued_iq()
                        with generation_lock:
                            item = (
                                generation[0], full_reset_pending[0], block
                            )
                        try:
                            iq_queue.put_nowait(item)
                        except queue.Full:
                            # DSP is behind. Preserve real-time operation by
                            # dropping stale IQ and starting one new stream
                            # generation instead of blocking socket reception.
                            with generation_lock:
                                generation[0] += 1
                                self.stats.queue_overruns += 1
                                item = (
                                    generation[0], full_reset_pending[0], block
                                )
                            discard_queued_iq()
                            iq_queue.put_nowait(item)
                except BaseException as exc:
                    reader_error.append(exc)
                finally:
                    reader_done.set()

            reader = threading.Thread(
                target=receive_iq,
                name="spyserver-iq-reader",
                daemon=True,
            )
            reader.start()
            observed_generation = generation[0]
            while not self.stop_requested:
                if duration is not None and time.monotonic() - started >= duration:
                    self.stop_requested = True
                    break
                try:
                    block_generation, full_reset, block = iq_queue.get(timeout=0.1)
                except queue.Empty:
                    if reader_done.is_set():
                        break
                    continue
                blocks = [block]
                # Batch the available run for vectorized DSP. A newer stream
                # generation makes all previously collected blocks stale.
                while True:
                    try:
                        next_generation, next_full_reset, next_block = (
                            iq_queue.get_nowait()
                        )
                    except queue.Empty:
                        break
                    if next_generation != block_generation:
                        block_generation = next_generation
                        full_reset = next_full_reset
                        blocks = [next_block]
                    else:
                        full_reset = full_reset or next_full_reset
                        blocks.append(next_block)
                if len(blocks) > 1:
                    block = np.concatenate(blocks)
                if self.stop_requested:
                    break
                self.iq.write(block)
                self.stats.iq_samples += len(block)
                if block_generation != observed_generation:
                    skipped_generations = block_generation - observed_generation
                    observed_generation = block_generation
                    if full_reset:
                        demodulator = LiveTetraDemodulator(
                            configuration.sample_rate,
                            channel_offset=(
                                configuration.frequency
                                - configuration.iq_center_frequency
                            ),
                            nominal_frequency=configuration.frequency,
                        )
                        reset_description = "full DSP reacquisition"
                        reset_prefix = "SpyServer IQ discontinuity"
                        with generation_lock:
                            full_reset_pending[0] = False
                    else:
                        retained = demodulator.recover_stream()
                        reset_description = (
                            "warm DSP recovery with carrier=%+.2f Hz" % retained
                            if retained is not None
                            else "DSP recovery; carrier acquisition still required"
                        )
                        reset_prefix = "Local IQ queue overrun"
                    if self.bridge is not None:
                        self.bridge.reset()
                    LOG.warning(
                        "%s: %s" % (reset_prefix, reset_description)
                        + (
                            " (%d discontinuities coalesced)" % skipped_generations
                            if skipped_generations > 1
                            else ""
                        )
                    )
                bursts, gap = demodulator.process(block)
                if gap and self.bridge is not None:
                    self.bridge.reset()
                confidences = demodulator.last_confidences
                for burst_index, burst in enumerate(bursts):
                    self.bits.write(burst)
                    if self.bridge is not None:
                        confidence = (
                            confidences[burst_index]
                            if burst_index < len(confidences)
                            else None
                        )
                        self.bridge.feed_burst(burst, confidence)
                    self.stats.bursts += 1
            if reader_error:
                raise reader_error[0]
            self.stats.sequence_gaps += client.sequence_gaps
        finally:
            try:
                client.stop()
            except (OSError, SpyServerDisconnected):
                pass
            client.close()

    def run(self, duration=None):
        try:
            while not self.stop_requested:
                try:
                    self._run_connection(duration=duration)
                    if duration is not None or self.stop_requested:
                        break
                    raise SpyServerDisconnected("SpyServer stream ended")
                except (OSError, SpyServerDisconnected) as exc:
                    if not self.reconnect:
                        raise
                    self.stats.reconnects += 1
                    LOG.warning(
                        "SpyServer connection lost: %s; reconnecting in %.1f s",
                        exc,
                        self.reconnect_delay,
                    )
                    if self.bridge is not None:
                        self.bridge.reset()
                    time.sleep(self.reconnect_delay)
        finally:
            self.bits.close()
            self.iq.close()
        return self.stats
