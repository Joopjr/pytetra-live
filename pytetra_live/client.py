"""Blocking, reconnectable SpyServer IQ client."""

from dataclasses import dataclass
import logging
import math
import socket
import time

import numpy as np

from . import protocol


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamConfiguration:
    frequency: int
    iq_center_frequency: int
    decimation: int
    sample_rate: float
    iq_format: int
    gain: int
    can_control: bool


class SpyServerDisconnected(ConnectionError):
    """Raised when the SpyServer closes or stalls the TCP stream."""


class SpyServerClient:
    def __init__(self, host, port, timeout=5.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket = None
        self.device_info = None
        self.client_sync = None
        self.last_sequence = {}
        self.sequence_gaps = 0
        self._last_gap_warning = 0.0
        self._suppressed_gap_warnings = 0

    def connect(self):
        self.close()
        self.socket = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        self.socket.settimeout(self.timeout)
        self._send(protocol.hello())
        self._wait_for_initial_state()
        return self

    def close(self):
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None

    def _send(self, data):
        if self.socket is None:
            raise SpyServerDisconnected("SpyServer is not connected")
        self.socket.sendall(data)

    def _recv_exact(self, size):
        data = bytearray()
        while len(data) < size:
            try:
                block = self.socket.recv(size - len(data))
            except socket.timeout as exc:
                raise SpyServerDisconnected("SpyServer receive timeout") from exc
            if not block:
                raise SpyServerDisconnected("SpyServer closed the connection")
            data.extend(block)
        return bytes(data)

    def receive_message(self):
        raw_header = self._recv_exact(protocol.MESSAGE_HEADER.size)
        header = protocol.MessageHeader.unpack(raw_header)
        body = self._recv_exact(header.body_size)
        # Status messages may use independent sequence counters even though they
        # share stream type zero. Only IQ discontinuities imply lost samples.
        if header.base_type in (
            protocol.MSG_UINT8_IQ,
            protocol.MSG_INT16_IQ,
            protocol.MSG_INT24_IQ,
            protocol.MSG_FLOAT_IQ,
        ):
            previous = self.last_sequence.get(header.stream_type)
            if (
                previous is not None
                and header.sequence_number != ((previous + 1) & 0xFFFFFFFF)
            ):
                self.sequence_gaps += 1
                now = time.monotonic()
                if now - self._last_gap_warning >= 0.5:
                    suffix = ""
                    if self._suppressed_gap_warnings:
                        suffix = " (%d additional gaps coalesced)" % (
                            self._suppressed_gap_warnings,
                        )
                    LOG.warning(
                        "SpyServer sequence gap: stream=%d expected=%d "
                        "received=%d%s",
                        header.stream_type,
                        (previous + 1) & 0xFFFFFFFF,
                        header.sequence_number,
                        suffix,
                    )
                    self._last_gap_warning = now
                    self._suppressed_gap_warnings = 0
                else:
                    self._suppressed_gap_warnings += 1
            self.last_sequence[header.stream_type] = header.sequence_number
        return header, body

    def _apply_status(self, header, body):
        if header.base_type == protocol.MSG_DEVICE_INFO:
            self.device_info = protocol.DeviceInfo.unpack(body)
            return True
        if header.base_type == protocol.MSG_CLIENT_SYNC:
            self.client_sync = protocol.ClientSync.unpack(body)
            return True
        return False

    def _wait_for_initial_state(self):
        deadline = time.monotonic() + self.timeout
        while self.device_info is None or self.client_sync is None:
            if time.monotonic() >= deadline:
                raise SpyServerDisconnected(
                    "SpyServer did not provide device information and client sync"
                )
            header, body = self.receive_message()
            self._apply_status(header, body)

    def set_setting(self, setting_type, value):
        self._send(protocol.setting(setting_type, value))

    def choose_decimation(self, minimum_rate=96000.0, maximum_rate=250000.0):
        info = self.device_info
        candidates = []
        for stage in range(info.minimum_iq_decimation, info.decimation_stage_count + 1):
            rate = info.sample_rate(stage)
            if rate >= minimum_rate and rate <= info.maximum_bandwidth:
                candidates.append((stage, rate))
        if not candidates:
            for stage in range(info.minimum_iq_decimation, info.decimation_stage_count + 1):
                rate = info.sample_rate(stage)
                if rate >= minimum_rate:
                    candidates.append((stage, rate))
        if not candidates:
            raise protocol.SpyServerProtocolError(
                "SpyServer cannot provide enough sample rate for TETRA"
            )
        preferred = [item for item in candidates if item[1] <= maximum_rate]
        return min(preferred or candidates, key=lambda item: item[1])

    def configure(self, frequency, center_frequency=None, gain=None, sample_rate=None):
        info = self.device_info
        sync = self.client_sync
        channel = int(frequency)
        requested_center = int(
            center_frequency if center_frequency is not None else channel
        )

        # IQ_FREQUENCY is a per-client digital tuning request. It does not move
        # the shared hardware centre while it remains inside the IQ range from
        # ClientSync, so it is also usable when device control is locked.
        if (
            sync.minimum_iq_center_frequency
            <= requested_center
            <= sync.maximum_iq_center_frequency
        ):
            center = requested_center
        elif sync.can_control:
            center = requested_center
        else:
            center = sync.iq_center_frequency
            LOG.warning(
                "Requested IQ center %d Hz is outside the locked SpyServer IQ "
                "range %d..%d Hz; retaining %d Hz and using local channel tuning",
                requested_center,
                sync.minimum_iq_center_frequency,
                sync.maximum_iq_center_frequency,
                center,
            )

        if not info.minimum_frequency <= center <= info.maximum_frequency:
            raise ValueError("Requested center frequency is outside device limits")

        # Normally SpyServer digitally centres this client's stream on the
        # channel. If that is impossible, request enough bandwidth for local
        # digital channel shifting instead.
        channel_guard = 25000.0
        required_rate = max(
            96000.0,
            2.0 * (abs(channel - center) + channel_guard),
        )

        if sample_rate is None:
            decimation, actual_rate = self.choose_decimation(
                minimum_rate=required_rate
            )
        else:
            stages = [
                (stage, info.sample_rate(stage))
                for stage in range(
                    info.minimum_iq_decimation,
                    info.decimation_stage_count + 1,
                )
            ]
            requested_rate = max(float(sample_rate), required_rate)
            safe = [item for item in stages if item[1] >= requested_rate]
            if not safe:
                raise ValueError("Requested sample rate is not supported")
            decimation, actual_rate = min(safe, key=lambda item: item[1])

        usable_bandwidth = min(actual_rate, float(info.maximum_bandwidth))
        if abs(channel - center) + channel_guard > usable_bandwidth / 2.0:
            raise ValueError("TETRA channel falls outside the requested IQ bandwidth")

        iq_format = info.forced_iq_format or protocol.FORMAT_INT16
        if iq_format not in (
            protocol.FORMAT_UINT8,
            protocol.FORMAT_INT16,
            protocol.FORMAT_FLOAT,
        ):
            raise protocol.SpyServerProtocolError(
                "SpyServer forced an unsupported IQ format"
            )

        selected_gain = sync.gain if gain is None else int(gain)
        if gain is not None:
            if not sync.can_control:
                LOG.warning("SpyServer control is locked; retaining gain %d", sync.gain)
                selected_gain = sync.gain
            elif not 0 <= selected_gain <= info.maximum_gain_index:
                raise ValueError("Gain is outside SpyServer device limits")

        self.set_setting(protocol.SETTING_IQ_FORMAT, iq_format)
        self.set_setting(protocol.SETTING_IQ_DECIMATION, decimation)
        self.set_setting(protocol.SETTING_IQ_FREQUENCY, center)
        self.set_setting(protocol.SETTING_STREAMING_MODE, protocol.STREAM_MODE_IQ_ONLY)
        if gain is not None and sync.can_control:
            self.set_setting(protocol.SETTING_GAIN, selected_gain)

        return StreamConfiguration(
            frequency=channel,
            iq_center_frequency=center,
            decimation=decimation,
            sample_rate=actual_rate,
            iq_format=iq_format,
            gain=selected_gain,
            can_control=sync.can_control,
        )

    def start(self):
        self.set_setting(protocol.SETTING_STREAMING_ENABLED, 1)

    def stop(self):
        if self.socket is not None:
            self.set_setting(protocol.SETTING_STREAMING_ENABLED, 0)

    @staticmethod
    def decode_iq(header, body):
        gain = math.pow(10.0, header.gain_db / 20.0)
        if header.base_type == protocol.MSG_UINT8_IQ:
            raw = np.frombuffer(body, dtype=np.uint8)
            if len(raw) % 2:
                raise protocol.SpyServerProtocolError("Odd UINT8 IQ body size")
            values = (raw.astype(np.float32) - 128.0) / (128.0 * gain)
        elif header.base_type == protocol.MSG_INT16_IQ:
            if len(body) % 4:
                raise protocol.SpyServerProtocolError("Invalid INT16 IQ body size")
            values = np.frombuffer(body, dtype="<i2").astype(np.float32)
            values /= 32768.0 * gain
        elif header.base_type == protocol.MSG_FLOAT_IQ:
            if len(body) % 8:
                raise protocol.SpyServerProtocolError("Invalid FLOAT IQ body size")
            values = np.frombuffer(body, dtype="<f4").astype(np.float32, copy=True)
            values /= gain
        elif header.base_type == protocol.MSG_INT24_IQ:
            raise protocol.SpyServerProtocolError("INT24 IQ is not supported")
        else:
            return None
        return (values[0::2] + 1j * values[1::2]).astype(np.complex64)

    def iq_blocks(self):
        while True:
            header, body = self.receive_message()
            if self._apply_status(header, body):
                continue
            iq = self.decode_iq(header, body)
            if iq is not None:
                yield iq
