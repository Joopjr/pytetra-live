"""Deferred cell-aware log-file creation."""

from datetime import datetime
import logging
from pathlib import Path
import re


CELL_PATTERN = re.compile(r"MCC\((\d+)\), MNC\((\d+)\), LA\((\d+)\)")


class CellLogHandler(logging.Handler):
    """Create a dated log after the first decoded cell identity is known."""

    MAX_BUFFERED_RECORDS = 1000

    def __init__(self, directory, debug=False):
        super().__init__(level=logging.DEBUG if debug else logging.INFO)
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.debug = bool(debug)
        self._buffer = []
        self._file_handler = None
        self._identity = None
        self._active_date = None
        self.path = None

    def _open(self, record, mcc=None, mnc=None, la=None):
        timestamp = datetime.fromtimestamp(record.created)
        if mcc is not None:
            self._identity = (mcc, mnc, la)
        elif self._identity is not None:
            mcc, mnc, la = self._identity
        identity = (
            f"MCC{mcc} MNC{mnc} LA{la}"
            if mcc is not None
            else "MCCunknown MNCunknown LAunknown"
        )
        suffix = " debug.log" if self.debug else ".log"
        self.path = self.directory / f"{timestamp:%Y-%m-%d} {identity}{suffix}"
        if self._file_handler is not None:
            self._file_handler.close()
        self._file_handler = logging.FileHandler(
            self.path, mode="a", encoding="utf-8"
        )
        self._file_handler.setLevel(self.level)
        self._file_handler.setFormatter(self.formatter)
        self._active_date = timestamp.date()
        for buffered_record in self._buffer:
            self._file_handler.emit(buffered_record)
        self._buffer.clear()

    def emit(self, record):
        try:
            if self._file_handler is None:
                match = CELL_PATTERN.search(record.getMessage())
                if match:
                    self._open(record, *match.groups())
                else:
                    self._buffer.append(record)
                    if len(self._buffer) > self.MAX_BUFFERED_RECORDS:
                        self._buffer.pop(0)
                    return
            record_date = datetime.fromtimestamp(record.created).date()
            if record_date != self._active_date:
                self._open(record)
            self._file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self):
        if self._file_handler is None and self._buffer:
            self._open(self._buffer[0])
        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None
        super().close()
