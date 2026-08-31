import logging
import queue
import unittest
import sys
import types
from unittest.mock import patch

import numpy as np

from pytetra_live.bridge import (
    PYTETRA_LEVEL,
    PyTetraBridge,
    QueuedPyTetraBridge,
    log_pytetra_output,
)


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
        elif kind == "burst_batch":
            hard, soft, count = payload
            received.append((kind, len(hard), len(soft), count))
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
            with patch("pytetra_live.bridge.LOG.log") as protocol:
                PyTetraBridge()
                FakeLogger.writer("DL; Layer 2 - MAC(MacResourcePdu)")

        protocol.assert_called_with(
            PYTETRA_LEVEL,
            "%s",
            "DL; Layer 2 - MAC(MacResourcePdu)",
        )

    def test_only_compact_layer2_and_layer3_use_pytetra_level(self):
        with patch("pytetra_live.bridge.LOG.log") as protocol, patch(
            "pytetra_live.bridge.LOG.info"
        ) as info:
            log_pytetra_output("DL; Cell; Layer 3 - MM(Example)")
            log_pytetra_output("UpperMac: MacResourcePdu(SSI(123))")
            log_pytetra_output("TETRA burst lock acquired")

        self.assertEqual(protocol.call_count, 2)
        protocol.assert_any_call(
            PYTETRA_LEVEL, "%s", "DL; Cell; Layer 3 - MM(Example)"
        )
        protocol.assert_any_call(
            PYTETRA_LEVEL, "%s", "UpperMac: MacResourcePdu(SSI(123))"
        )
        info.assert_called_once_with("%s", "TETRA burst lock acquired")
        self.assertEqual(logging.getLevelName(PYTETRA_LEVEL), "PYTETRA")

    def test_queue_overrun_cannot_escape_as_queue_full(self):
        bridge = object.__new__(QueuedPyTetraBridge)
        bridge.events = queue.Queue(maxsize=1)
        bridge.events.put_nowait(("burst", None))
        bridge.overruns = 0
        bridge._last_warning = 0.0
        bridge._reset_pending = False

        with patch.object(bridge, "_raise_worker_error"), patch(
            "pytetra_live.bridge.LOG.warning"
        ):
            bridge._put(("burst", (b"bits", None)))

        self.assertEqual(bridge.overruns, 1)
        self.assertEqual(bridge.events.get_nowait(), ("reset", None))

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

    def test_burst_batch_uses_one_ordered_queue_event(self):
        with patch("pytetra_live.bridge.LOG.info") as info:
            bridge = QueuedPyTetraBridge(
                capacity=8,
                _worker_target=recording_decoder_process,
            )
            bursts = np.zeros((3, 510), dtype=np.uint8)
            confidence = np.ones((3, 510), dtype=np.float32)
            bridge.feed_bursts(bursts, confidence)
            bridge.close()

        output = "\n".join(str(call) for call in info.call_args_list)
        self.assertIn("('burst_batch', 1530, 6120, 3)", output)


if __name__ == "__main__":
    unittest.main()
