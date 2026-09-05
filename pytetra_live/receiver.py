"""Orchestrate SpyServer acquisition, demodulation, and PyTetra delivery."""

from dataclasses import dataclass
import logging
import queue
import threading
import time

import numpy as np

from .bridge import BitFileSink, IqFileSink, QueuedPyTetraBridge
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
    decoder_overruns: int = 0


class LiveReceiver:
    # Keep enough elasticity for short scheduler stalls without accumulating
    # seconds of latency when the decoder cannot sustain real-time operation.
    IQ_QUEUE_BLOCKS = 128
    DSP_BATCH_SECONDS = 0.25

    def __init__(
        self,
        host,
        port,
        frequency,
        center_frequency=None,
        gain=None,
        sample_rate=None,
        debug=False,
        show_esi=False,
        show_telemetry=False,
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
        self.telemetry_interval = (
            float(show_telemetry)
            if show_telemetry is not None
            else (30.0 if debug else None)
        )
        self.bits = BitFileSink(bits_output)
        self.iq = IqFileSink(iq_output)
        self.bridge = (
            QueuedPyTetraBridge(
                debug=debug,
                show_esi=show_esi,
                carrier_frequency=self.frequency,
                output_observer=self._observe_pytetra_output,
            )
            if decode
            else None
        )
        self.reconnect = bool(reconnect)
        self.reconnect_delay = float(reconnect_delay)
        self.timeout = float(timeout)
        self.stats = ReceiverStats()
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def _observe_pytetra_output(self, message):
        self.bits.observe_pytetra_output(message)
        self.iq.observe_pytetra_output(message)

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
            generation_lock = threading.Lock()

            def discard_queued_iq():
                while True:
                    try:
                        iq_queue.get_nowait()
                    except queue.Empty:
                        return

            def receive_iq():
                observed_discontinuities = client.sequence_discontinuities
                try:
                    for block in client.iq_blocks():
                        if self.stop_requested:
                            break
                        if client.sequence_discontinuities != observed_discontinuities:
                            observed_discontinuities = client.sequence_discontinuities
                            with generation_lock:
                                generation[0] += 1
                            discard_queued_iq()
                        with generation_lock:
                            item = (
                                generation[0],
                                client.sequence_discontinuities,
                                block,
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
                                    generation[0],
                                    client.sequence_discontinuities,
                                    block,
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
            processed_discontinuities = 0
            next_performance_log = (
                time.monotonic() + self.telemetry_interval
                if self.telemetry_interval is not None else float("inf")
            )
            previous_accepted = 0
            previous_rejected = 0
            previous_training_errors = 0
            metrics_framer = demodulator.framer
            while not self.stop_requested:
                if duration is not None and time.monotonic() - started >= duration:
                    self.stop_requested = True
                    break
                try:
                    block_generation, server_gaps, block = iq_queue.get(timeout=0.1)
                except queue.Empty:
                    if reader_done.is_set():
                        break
                    continue
                blocks = [block]
                batch_samples = len(block)
                maximum_batch_samples = max(
                    1,
                    int(configuration.sample_rate * self.DSP_BATCH_SECONDS),
                )
                # Batch the available run for vectorized DSP. A newer stream
                # generation makes all previously collected blocks stale.
                while True:
                    try:
                        next_generation, next_server_gaps, next_block = (
                            iq_queue.get_nowait()
                        )
                    except queue.Empty:
                        break
                    if next_generation != block_generation:
                        block_generation = next_generation
                        server_gaps = next_server_gaps
                        blocks = [next_block]
                        batch_samples = len(next_block)
                    else:
                        server_gaps = max(server_gaps, next_server_gaps)
                        blocks.append(next_block)
                        batch_samples += len(next_block)
                    if batch_samples >= maximum_batch_samples:
                        break
                if len(blocks) > 1:
                    block = np.concatenate(blocks)
                if self.stop_requested:
                    break
                self.iq.write(block)
                self.stats.iq_samples += len(block)
                if block_generation != observed_generation:
                    skipped_generations = block_generation - observed_generation
                    observed_generation = block_generation
                    if server_gaps != processed_discontinuities:
                        processed_discontinuities = server_gaps
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
                    self.stats.bursts += 1
                if self.bridge is not None and bursts:
                    confidence_batch = (
                        confidences if len(confidences) == len(bursts) else None
                    )
                    self.bridge.feed_bursts(bursts, confidence_batch)
                now = time.monotonic()
                if now >= next_performance_log:
                    dsp = demodulator.stats
                    if demodulator.framer is not metrics_framer:
                        metrics_framer = demodulator.framer
                        previous_accepted = 0
                        previous_rejected = 0
                        previous_training_errors = 0
                    input_seconds = dsp.input_samples / configuration.sample_rate
                    realtime = (
                        input_seconds / dsp.processing_seconds
                        if dsp.processing_seconds > 0.0
                        else 0.0
                    )
                    total_stages = max(dsp.processing_seconds, 1e-12)
                    LOG.info(
                        "Performance: dsp_realtime=%.2fx iq_queue=%d/%d "
                        "decoder_queue_batches=%d decoder_queue_bursts=%d "
                        "decoder_overruns=%d decoder_dropped_bursts=%d stages="
                        "channel(%.1f%%),resample(%.1f%%),filter(%.1f%%),carrier(%.1f%%),"
                        "timing(%.1f%%),framing(%.1f%%)",
                        realtime,
                        iq_queue.qsize(),
                        self.IQ_QUEUE_BLOCKS,
                        self.bridge.queue_depth if self.bridge is not None else 0,
                        self.bridge.queue_bursts if self.bridge is not None else 0,
                        self.bridge.overruns if self.bridge is not None else 0,
                        self.bridge.dropped_bursts if self.bridge is not None else 0,
                        100.0 * dsp.channel_filter_seconds / total_stages,
                        100.0 * dsp.resample_seconds / total_stages,
                        100.0 * dsp.filter_seconds / total_stages,
                        100.0 * dsp.carrier_seconds / total_stages,
                        100.0 * dsp.timing_seconds / total_stages,
                        100.0 * dsp.framing_seconds / total_stages,
                    )
                    accepted_total = demodulator.framer.accepted
                    rejected_total = demodulator.framer.rejected
                    training_total = demodulator.framer.training_error_sum
                    accepted = max(0, accepted_total - previous_accepted)
                    rejected = max(0, rejected_total - previous_rejected)
                    training_delta = max(0, training_total - previous_training_errors)
                    previous_accepted = accepted_total
                    previous_rejected = rejected_total
                    previous_training_errors = training_total
                    examined = accepted + rejected
                    burst_success = 100.0 * accepted / examined if examined else 0.0
                    training_errors = (
                        training_delta / accepted
                        if accepted else 0.0
                    )
                    carrier = demodulator.carrier
                    timing_ppm = 1e6 * (
                        demodulator.timing.omega / demodulator.timing.omega_nominal - 1.0
                    )
                    LOG.info(
                        "Signal quality: input_level=%.1f dBFS estimated_snr=%.1f dB "
                        "carrier_coherence=%.3f frequency_error=%+.2f Hz "
                        "phase_error_rms=%.1f deg timing_error_rms=%.4f "
                        "samples_per_symbol=%.6f timing_drift=%+.1f ppm "
                        "burst_success=%.1f%% training_errors=%.2f lock=%s",
                        dsp.input_level_dbfs,
                        dsp.estimated_snr_db,
                        carrier.coherence if carrier is not None else 0.0,
                        carrier.frequency if carrier is not None else 0.0,
                        dsp.phase_error_rms_degrees,
                        demodulator.timing.error_rms,
                        demodulator.timing.omega,
                        timing_ppm,
                        burst_success,
                        training_errors,
                        demodulator.framer.locked,
                    )
                    next_performance_log = now + self.telemetry_interval
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
            try:
                if self.bridge is not None:
                    self.stats.decoder_overruns = self.bridge.overruns
                    self.bridge.close()
            finally:
                self.bits.close()
                self.iq.close()
        return self.stats
