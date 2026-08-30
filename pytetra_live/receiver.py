"""Orchestrate SpyServer acquisition, demodulation, and PyTetra delivery."""

from dataclasses import dataclass
import logging
import time

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


class LiveReceiver:
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
            observed_sequence_gaps = client.sequence_gaps
            for block in client.iq_blocks():
                if self.stop_requested:
                    break
                if duration is not None and time.monotonic() - started >= duration:
                    self.stop_requested = True
                    break
                self.iq.write(block)
                self.stats.iq_samples += len(block)
                if client.sequence_gaps != observed_sequence_gaps:
                    observed_sequence_gaps = client.sequence_gaps
                    demodulator = LiveTetraDemodulator(
                        configuration.sample_rate,
                        channel_offset=(
                            configuration.frequency
                            - configuration.iq_center_frequency
                        ),
                        nominal_frequency=configuration.frequency,
                    )
                    if self.bridge is not None:
                        self.bridge.reset()
                    LOG.warning(
                        "DSP state reset after a SpyServer sequence gap"
                    )
                bursts, gap = demodulator.process(block)
                if gap and self.bridge is not None:
                    self.bridge.reset()
                for burst in bursts:
                    self.bits.write(burst)
                    if self.bridge is not None:
                        self.bridge.feed_burst(burst)
                    self.stats.bursts += 1
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
