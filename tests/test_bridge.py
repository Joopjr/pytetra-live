import unittest
import sys
import types
from unittest.mock import patch

import numpy as np

from pytetra_live.bridge import PyTetraBridge, QueuedPyTetraBridge


def recording_decoder_process(events, output, errors, debug):
    """Picklable CI worker that verifies ordering and serialized payloads."""
    received = []
    while True:
        kind, payload = events.get()
        if kind == "stop":
            output.put("events=%s" % received)
            output.put(None)
            return
        if kind == "burst":
            hard, soft = payload
            received.append((kind, list(hard), len(soft)))
        else:
            received.append((kind, None, None))


class QueuedBridgeTestCase(unittest.TestCase):
    def test_pytetra_lines_are_routed_through_timestamped_logging(self):
        class FakeLogger:
            writer = None

            @classmethod
            def set_writer(cls, writer):
                cls.writer = writer

        class FakeStack:
            def __init__(self, user_layer, debug=False):
                self.user_layer = user_layer
                self.debug = debug

        fake_cli = types.ModuleType("pytetra.cli")
        fake_cli.ConsoleUserLayer = object
        fake_logger = types.ModuleType("pytetra.logger")
        fake_logger.Logger = FakeLogger
        fake_stack = types.ModuleType("pytetra.stack")
        fake_stack.TetraStack = FakeStack
        fake_package = types.ModuleType("pytetra")
        modules = {
            "pytetra": fake_package,
            "pytetra.cli": fake_cli,
            "pytetra.logger": fake_logger,
            "pytetra.stack": fake_stack,
        }

        with patch.dict(sys.modules, modules):
            with patch("pytetra_live.bridge.LOG.info") as info:
                PyTetraBridge()
                FakeLogger.writer("DL; Layer 2 - MAC(MacResourcePdu)")

        info.assert_called_with("%s", "DL; Layer 2 - MAC(MacResourcePdu)")

    def test_decoder_process_accepts_ordered_copied_events(self):
        with patch("pytetra_live.bridge.LOG.info") as info:
            bridge = QueuedPyTetraBridge(
                capacity=8,
                _worker_target=recording_decoder_process,
            )
            burst = np.asarray([0, 1, 1, 0], dtype=np.uint8)
            confidence = np.asarray([-1.0, 0.8, 0.7, -0.9], dtype=np.float32)
            bridge.feed_burst(burst, confidence)
            bridge.reset()
            burst[:] = 0
            confidence[:] = 0
            bridge.close()

        output = "\n".join(str(call) for call in info.call_args_list)
        self.assertIn("('burst', [0, 1, 1, 0], 16)", output)
        self.assertIn("('reset', None, None)", output)
        self.assertEqual(bridge.worker.exitcode, 0)
        self.assertEqual(bridge.overruns, 0)


if __name__ == "__main__":
    unittest.main()
