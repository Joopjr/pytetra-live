import logging
from pathlib import Path
import tempfile
import unittest

from pytetra_live.logfiles import CellLogHandler


class CellLogHandlerTestCase(unittest.TestCase):
    def test_filename_uses_first_decoded_cell_and_keeps_earlier_output(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = CellLogHandler(directory)
            handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            logger = logging.getLogger("test.cell-log")
            logger.handlers = [handler]
            logger.propagate = False
            logger.setLevel(logging.DEBUG)

            logger.info("SpyServer connected")
            logger.info(
                "DL; MCC(204), MNC(1000), LA(2333); Layer 2 - MAC(Example)"
            )
            handler.close()

            files = list(Path(directory).glob("*.log"))
            self.assertEqual(len(files), 1)
            self.assertRegex(files[0].name, r"^\d{4}-\d{2}-\d{2} MCC204 MNC1000 LA2333\.log$")
            contents = files[0].read_text(encoding="utf-8")
            self.assertIn("SpyServer connected", contents)
            self.assertIn("MCC(204), MNC(1000), LA(2333)", contents)

    def test_debug_filename_and_level(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = CellLogHandler(directory, debug=True)
            handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            record = logging.LogRecord(
                "test", logging.DEBUG, __file__, 1,
                "MCC(204), MNC(1000), LA(2333)", (), None,
            )
            handler.emit(record)
            handler.close()
            files = list(Path(directory).glob("* debug.log"))
            self.assertEqual(len(files), 1)
            self.assertIn("DEBUG", files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
