from datetime import datetime
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

    def test_long_running_log_rotates_at_local_midnight(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = CellLogHandler(directory)
            handler.setFormatter(logging.Formatter("%(message)s"))

            before_midnight = logging.LogRecord(
                "test", logging.INFO, __file__, 1,
                "DL; MCC(204), MNC(1000), LA(2333); before midnight", (), None,
            )
            before_midnight.created = datetime(2026, 9, 1, 23, 59, 59).timestamp()
            after_midnight = logging.LogRecord(
                "test", logging.INFO, __file__, 1,
                "after midnight", (), None,
            )
            after_midnight.created = datetime(2026, 9, 2, 0, 0, 1).timestamp()

            handler.emit(before_midnight)
            handler.emit(after_midnight)
            handler.close()

            first = Path(directory) / "2026-09-01 MCC204 MNC1000 LA2333.log"
            second = Path(directory) / "2026-09-02 MCC204 MNC1000 LA2333.log"
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertIn("before midnight", first.read_text(encoding="utf-8"))
            self.assertNotIn("after midnight", first.read_text(encoding="utf-8"))
            self.assertIn("after midnight", second.read_text(encoding="utf-8"))

    def test_deleted_active_log_is_recreated(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = CellLogHandler(directory)
            handler.setFormatter(logging.Formatter("%(message)s"))
            first = logging.LogRecord(
                "test", logging.INFO, __file__, 1,
                "DL; MCC(204), MNC(1000), LA(2333); first", (), None,
            )
            second = logging.LogRecord(
                "test", logging.INFO, __file__, 1, "second", (), None,
            )

            handler.emit(first)
            path = handler.path
            path.unlink()
            handler.emit(second)
            handler.close()

            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "second")

    def test_replaced_active_log_is_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = CellLogHandler(directory)
            handler.setFormatter(logging.Formatter("%(message)s"))
            first = logging.LogRecord(
                "test", logging.INFO, __file__, 1,
                "DL; MCC(204), MNC(1000), LA(2333); first", (), None,
            )
            second = logging.LogRecord(
                "test", logging.INFO, __file__, 1, "second", (), None,
            )

            handler.emit(first)
            path = handler.path
            rotated = path.with_suffix(".log.old")
            path.rename(rotated)
            path.write_text("replacement\n", encoding="utf-8")
            handler.emit(second)
            handler.close()

            self.assertEqual(
                path.read_text(encoding="utf-8"), "replacement\nsecond\n"
            )
            self.assertIn("first", rotated.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
