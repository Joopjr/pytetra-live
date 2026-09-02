import socket
import struct
import threading
import unittest
from unittest.mock import patch

import numpy as np

from pytetra_live import protocol
from pytetra_live.client import SpyServerClient


def message(message_type, stream_type, sequence, body, flags=0):
    header = protocol.MESSAGE_HEADER.pack(
        protocol.PROTOCOL_ID,
        message_type | (flags << 16),
        stream_type,
        sequence,
        len(body),
    )
    return header + body


class FakeSpyServer(threading.Thread):
    def __init__(self, listener):
        super().__init__(daemon=True)
        self.listener = listener
        self.commands = []
        self.error = None

    @staticmethod
    def recv_exact(connection, size):
        data = bytearray()
        while len(data) < size:
            block = connection.recv(size - len(data))
            if not block:
                raise ConnectionError("client disconnected")
            data.extend(block)
        return bytes(data)

    def receive_command(self, connection):
        header = self.recv_exact(connection, protocol.COMMAND_HEADER.size)
        command_type, size = protocol.COMMAND_HEADER.unpack(header)
        body = self.recv_exact(connection, size)
        self.commands.append((command_type, body))
        return command_type, body

    def run(self):
        try:
            connection, unused_address = self.listener.accept()
            with connection:
                self.receive_command(connection)
                device = protocol.DEVICE_INFO_BODY.pack(
                    1, 1, 10000000, 1000000, 8, 1, 21,
                    24000000, 1800000000, 12, 4, 0,
                )
                sync = protocol.CLIENT_SYNC_BODY.pack(
                    1, 18, 412475000, 412475000, 412475000,
                    411975000, 412975000, 411975000, 412975000,
                )
                first = message(protocol.MSG_DEVICE_INFO, 0, 0, device)
                second = message(protocol.MSG_CLIENT_SYNC, 0, 1, sync)
                for offset in range(0, len(first), 3):
                    connection.sendall(first[offset:offset + 3])
                connection.sendall(second)
                for unused in range(4):
                    self.receive_command(connection)
                self.receive_command(connection)
                iq = np.asarray([1000, -1000, 2000, -2000], dtype="<i2").tobytes()
                connection.sendall(message(protocol.MSG_INT16_IQ, 1, 0, iq))
        except OSError as exc:
            # Tests that exercise only local client helpers do not connect;
            # closing their listener is the normal way to stop this thread.
            if self.listener.fileno() != -1:
                self.error = exc
        except Exception as exc:
            self.error = exc


class ClientTestCase(unittest.TestCase):
    def setUp(self):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.server = FakeSpyServer(self.listener)
        self.server.start()

    def tearDown(self):
        self.listener.close()
        self.server.join(timeout=2)
        if self.server.error is not None:
            raise self.server.error

    def test_connect_configure_and_receive_int16(self):
        host, port = self.listener.getsockname()
        client = SpyServerClient(host, port, timeout=2).connect()
        configuration = client.configure(412475000)
        self.assertEqual(configuration.iq_center_frequency, 412475000)
        self.assertEqual(configuration.sample_rate, 156250.0)
        self.assertEqual(configuration.gain, 18)
        client.start()
        block = next(client.iq_blocks())
        np.testing.assert_allclose(
            block,
            np.asarray([1000 - 1000j, 2000 - 2000j]) / 32768.0,
            rtol=1e-6,
        )
        client.close()
        self.assertEqual(self.server.commands[0][0], protocol.CMD_HELLO)
        settings = [
            protocol.SETTING_BODY.unpack(body)
            for command_type, body in self.server.commands[1:]
            if command_type == protocol.CMD_SET_SETTING
        ]
        self.assertIn((protocol.SETTING_IQ_DECIMATION, 6), settings)
        self.assertIn((protocol.SETTING_IQ_FREQUENCY, 412475000), settings)
        self.assertNotIn((protocol.SETTING_GAIN, 18), settings)

    def test_iq_format_decoders(self):
        uint_header = protocol.MessageHeader(0, protocol.MSG_UINT8_IQ, 1, 0, 4)
        uint_iq = SpyServerClient.decode_iq(uint_header, bytes([128, 128, 255, 0]))
        np.testing.assert_allclose(uint_iq[0], 0j)

        float_header = protocol.MessageHeader(0, protocol.MSG_FLOAT_IQ, 1, 0, 8)
        float_iq = SpyServerClient.decode_iq(
            float_header, struct.pack("<ff", 0.25, -0.5)
        )
        np.testing.assert_allclose(float_iq, [0.25 - 0.5j])

    def test_locked_device_uses_per_client_iq_tuning(self):
        client = SpyServerClient("unused", 5556)
        client.device_info = protocol.DeviceInfo(
            1, 1, 6000000, 4800000, 11, 1, 21,
            24000000, 1800000000, 12, 0, 0,
        )
        client.client_sync = protocol.ClientSync(
            False, 18, 412475000, 412475000, 412475000,
            410075000, 414875000, 410075000, 414875000,
        )
        settings = []
        client.set_setting = lambda setting, value: settings.append((setting, value))

        configuration = client.configure(413462500)

        self.assertEqual(configuration.iq_center_frequency, 413462500)
        self.assertEqual(configuration.decimation, 6)
        self.assertEqual(configuration.sample_rate, 93750.0)
        self.assertIn((protocol.SETTING_IQ_DECIMATION, 6), settings)
        self.assertIn((protocol.SETTING_IQ_FREQUENCY, 413462500), settings)

    def test_locked_device_falls_back_to_wide_local_tuning(self):
        client = SpyServerClient("unused", 5556)
        client.device_info = protocol.DeviceInfo(
            1, 1, 6000000, 4800000, 11, 1, 21,
            24000000, 1800000000, 12, 0, 0,
        )
        client.client_sync = protocol.ClientSync(
            False, 18, 412475000, 412475000, 412475000,
            412000000, 413000000, 412000000, 413000000,
        )
        settings = []
        client.set_setting = lambda setting, value: settings.append((setting, value))

        configuration = client.configure(413462500)

        self.assertEqual(configuration.iq_center_frequency, 412475000)
        self.assertEqual(configuration.decimation, 1)
        self.assertEqual(configuration.sample_rate, 3000000.0)
        self.assertIn((protocol.SETTING_IQ_DECIMATION, 1), settings)

    def test_status_sequence_numbers_do_not_count_as_iq_gaps(self):
        client = SpyServerClient("unused", 5556)
        client.socket, sender = socket.socketpair()
        try:
            sender.sendall(message(protocol.MSG_DEVICE_INFO, 0, 0, b""))
            sender.sendall(message(protocol.MSG_CLIENT_SYNC, 0, 0, b""))
            client.receive_message()
            client.receive_message()
            self.assertEqual(client.sequence_gaps, 0)
            self.assertEqual(client.last_sequence, {})
        finally:
            sender.close()
            client.close()

    def test_adjacent_iq_gaps_form_one_discontinuity(self):
        client = SpyServerClient("unused", 5556)
        client.socket, sender = socket.socketpair()
        try:
            sender.sendall(message(protocol.MSG_INT16_IQ, 1, 0, b""))
            sender.sendall(message(protocol.MSG_INT16_IQ, 1, 2, b""))
            sender.sendall(message(protocol.MSG_INT16_IQ, 1, 4, b""))
            with patch("pytetra_live.client.time.monotonic", side_effect=[1.0, 1.1]):
                client.receive_message()
                client.receive_message()
                client.receive_message()
            self.assertEqual(client.sequence_gaps, 2)
            self.assertEqual(client.sequence_discontinuities, 1)
        finally:
            sender.close()
            client.close()


if __name__ == "__main__":
    unittest.main()
