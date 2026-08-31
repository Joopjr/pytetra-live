import unittest
from unittest.mock import patch

import numpy as np

from pytetra_live.bridge import PyTetraBridge, QueuedPyTetraBridge


class RecordingBridge:
    instances = []

    def __init__(self, debug=False):
        self.events = []
        self.instances.append(self)

    def reset(self):
        self.events.append(("reset", None))

    def feed_burst(self, burst, confidence=None):
        self.events.append(
            ("burst", (np.asarray(burst).copy(), np.asarray(confidence).copy()))
        )


class QueuedBridgeTestCase(unittest.TestCase):
    def test_pytetra_lines_are_routed_through_timestamped_logging(self):
        from pytetra.logger import Logger

        with patch("pytetra_live.bridge.LOG.info") as info:
            PyTetraBridge()
            Logger.log("DL; Layer 2 - MAC(MacResourcePdu)")
        Logger.set_writer(None)

        info.assert_called_with("%s", "DL; Layer 2 - MAC(MacResourcePdu)")

    def test_decoder_worker_preserves_event_order_and_copies_input(self):
        RecordingBridge.instances = []
        with patch("pytetra_live.bridge.PyTetraBridge", RecordingBridge):
            bridge = QueuedPyTetraBridge(capacity=8)
            burst = np.asarray([0, 1, 1, 0], dtype=np.uint8)
            confidence = np.asarray([-1.0, 0.8, 0.7, -0.9], dtype=np.float32)
            bridge.feed_burst(burst, confidence)
            bridge.reset()
            burst[:] = 0
            confidence[:] = 0
            bridge.close()

        events = RecordingBridge.instances[0].events
        self.assertEqual([event[0] for event in events], ["burst", "reset"])
        np.testing.assert_array_equal(events[0][1][0], [0, 1, 1, 0])
        np.testing.assert_allclose(events[0][1][1], [-1.0, 0.8, 0.7, -0.9])


if __name__ == "__main__":
    unittest.main()
