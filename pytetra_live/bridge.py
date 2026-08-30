"""Optional direct bridge from validated bursts into PyTetra."""

import logging

import numpy as np


LOG = logging.getLogger(__name__)


class PyTetraUnavailable(RuntimeError):
    pass


class PyTetraBridge:
    def __init__(self, debug=False):
        self.debug = bool(debug)
        self.stack = None
        self.resets = 0
        self._create_stack()

    def _create_stack(self):
        try:
            from pytetra.cli import ConsoleUserLayer
            from pytetra.stack import TetraStack
        except ImportError as exc:
            raise PyTetraUnavailable(
                "PyTetra is not importable. Install the sibling pytetra project "
                "or run PyTetra-live with --no-decode."
            ) from exc
        self.stack = TetraStack(ConsoleUserLayer, debug=self.debug)

    def reset(self):
        self.resets += 1
        self._create_stack()
        LOG.info("PyTetra stream state reset after a demodulator gap")

    def feed_burst(self, burst):
        self.stack.phy.feed([int(bit) for bit in burst])


class BitFileSink:
    def __init__(self, path=None):
        self.path = path
        self.file = open(path, "ab", buffering=0) if path else None
        self.bits_written = 0

    def write(self, burst):
        if self.file is not None:
            self.file.write(bytes(int(bit) for bit in burst))
        self.bits_written += len(burst)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class IqFileSink:
    """Write normalized interleaved float32 IQ for reproducible diagnostics."""

    def __init__(self, path=None):
        self.path = path
        self.file = open(path, "ab", buffering=0) if path else None
        self.samples_written = 0

    def write(self, iq):
        if self.file is not None:
            interleaved = np.empty(len(iq) * 2, dtype="<f4")
            interleaved[0::2] = iq.real
            interleaved[1::2] = iq.imag
            self.file.write(interleaved.tobytes())
        self.samples_written += len(iq)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
