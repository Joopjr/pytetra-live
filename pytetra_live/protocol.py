"""Minimal SpyServer protocol definitions used by the IQ client."""

from dataclasses import dataclass
import struct


PROTOCOL_VERSION = (2 << 24) | 1700
PROTOCOL_ID = 0x53505953
MAX_COMMAND_BODY = 256
MAX_MESSAGE_BODY = 1 << 20

CMD_HELLO = 0
CMD_SET_SETTING = 2
CMD_PING = 3

SETTING_STREAMING_MODE = 0
SETTING_STREAMING_ENABLED = 1
SETTING_GAIN = 2
SETTING_IQ_FORMAT = 100
SETTING_IQ_FREQUENCY = 101
SETTING_IQ_DECIMATION = 102
SETTING_IQ_DIGITAL_GAIN = 103

STREAM_TYPE_STATUS = 0
STREAM_TYPE_IQ = 1
STREAM_MODE_IQ_ONLY = STREAM_TYPE_IQ

FORMAT_UINT8 = 1
FORMAT_INT16 = 2
FORMAT_INT24 = 3
FORMAT_FLOAT = 4

MSG_DEVICE_INFO = 0
MSG_CLIENT_SYNC = 1
MSG_PONG = 2
MSG_READ_SETTING = 3
MSG_UINT8_IQ = 100
MSG_INT16_IQ = 101
MSG_INT24_IQ = 102
MSG_FLOAT_IQ = 103

COMMAND_HEADER = struct.Struct("<II")
MESSAGE_HEADER = struct.Struct("<IIIII")
SETTING_BODY = struct.Struct("<II")
DEVICE_INFO_BODY = struct.Struct("<12I")
CLIENT_SYNC_BODY = struct.Struct("<9I")


class SpyServerProtocolError(RuntimeError):
    """Raised for malformed or unsupported SpyServer protocol data."""


@dataclass(frozen=True)
class MessageHeader:
    protocol_id: int
    message_type: int
    stream_type: int
    sequence_number: int
    body_size: int

    @property
    def base_type(self):
        return self.message_type & 0xFFFF

    @property
    def gain_db(self):
        return (self.message_type >> 16) & 0xFFFF

    @classmethod
    def unpack(cls, data):
        if len(data) != MESSAGE_HEADER.size:
            raise SpyServerProtocolError("Invalid SpyServer message header size")
        header = cls(*MESSAGE_HEADER.unpack(data))
        if header.body_size > MAX_MESSAGE_BODY:
            raise SpyServerProtocolError(
                "SpyServer message body exceeds the 1 MiB protocol limit"
            )
        return header


@dataclass(frozen=True)
class DeviceInfo:
    device_type: int
    device_serial: int
    maximum_sample_rate: int
    maximum_bandwidth: int
    decimation_stage_count: int
    gain_stage_count: int
    maximum_gain_index: int
    minimum_frequency: int
    maximum_frequency: int
    resolution: int
    minimum_iq_decimation: int
    forced_iq_format: int

    @classmethod
    def unpack(cls, data):
        if len(data) < DEVICE_INFO_BODY.size:
            raise SpyServerProtocolError("Truncated SpyServer device information")
        return cls(*DEVICE_INFO_BODY.unpack_from(data))

    def sample_rate(self, decimation):
        if not self.minimum_iq_decimation <= decimation <= self.decimation_stage_count:
            raise ValueError("Unsupported SpyServer IQ decimation stage")
        return self.maximum_sample_rate / float(1 << decimation)


@dataclass(frozen=True)
class ClientSync:
    can_control: bool
    gain: int
    device_center_frequency: int
    iq_center_frequency: int
    fft_center_frequency: int
    minimum_iq_center_frequency: int
    maximum_iq_center_frequency: int
    minimum_fft_center_frequency: int
    maximum_fft_center_frequency: int

    @classmethod
    def unpack(cls, data):
        if len(data) < CLIENT_SYNC_BODY.size:
            raise SpyServerProtocolError("Truncated SpyServer client synchronization")
        values = list(CLIENT_SYNC_BODY.unpack_from(data))
        values[0] = bool(values[0])
        return cls(*values)


def command(command_type, body=b""):
    if len(body) > MAX_COMMAND_BODY:
        raise ValueError("SpyServer command body exceeds protocol limit")
    return COMMAND_HEADER.pack(command_type, len(body)) + body


def hello(application_name="PyTetra-live"):
    body = struct.pack("<I", PROTOCOL_VERSION) + application_name.encode("ascii")
    return command(CMD_HELLO, body)


def setting(setting_type, value):
    return command(CMD_SET_SETTING, SETTING_BODY.pack(setting_type, int(value)))

