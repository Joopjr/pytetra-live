import struct
import unittest

from pytetra_live import protocol


class ProtocolTestCase(unittest.TestCase):
    def test_hello_contains_version_and_name(self):
        packet = protocol.hello("test")
        command_type, body_size = protocol.COMMAND_HEADER.unpack_from(packet)
        self.assertEqual(command_type, protocol.CMD_HELLO)
        self.assertEqual(body_size, 8)
        self.assertEqual(struct.unpack_from("<I", packet, 8)[0], protocol.PROTOCOL_VERSION)
        self.assertEqual(packet[12:], b"test")

    def test_setting_is_little_endian(self):
        packet = protocol.setting(protocol.SETTING_IQ_FREQUENCY, 392475000)
        self.assertEqual(protocol.COMMAND_HEADER.unpack_from(packet), (2, 8))
        self.assertEqual(
            protocol.SETTING_BODY.unpack_from(packet, protocol.COMMAND_HEADER.size),
            (protocol.SETTING_IQ_FREQUENCY, 392475000),
        )

    def test_oversized_message_is_rejected(self):
        raw = protocol.MESSAGE_HEADER.pack(0, 100, 1, 0, protocol.MAX_MESSAGE_BODY + 1)
        with self.assertRaises(protocol.SpyServerProtocolError):
            protocol.MessageHeader.unpack(raw)

    def test_device_sample_rates(self):
        values = (1, 2, 10000000, 1000000, 8, 1, 21, 24000000, 1800000000, 12, 4, 0)
        info = protocol.DeviceInfo.unpack(protocol.DEVICE_INFO_BODY.pack(*values))
        self.assertEqual(info.sample_rate(6), 156250.0)
        with self.assertRaises(ValueError):
            info.sample_rate(9)


if __name__ == "__main__":
    unittest.main()

