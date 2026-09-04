import unittest
import argparse
import contextlib
import io

from pytetra_live.cli import (
    DEFAULT_LOG_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    build_argument_parser,
    telemetry_interval,
)


class CliTestCase(unittest.TestCase):
    def test_required_connection_arguments(self):
        parser = build_argument_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
        args = parser.parse_args(
            ["--host", "127.0.0.1", "--port", "5556", "--frequency", "412475000"]
        )
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 5556)
        self.assertEqual(args.frequency, 412475000)
        self.assertIsNone(args.gain)
        self.assertIsNone(args.center_frequency)
        self.assertFalse(args.show_esi)
        self.assertIsNone(args.show_telemetry)

    def test_optional_protocol_output_arguments(self):
        parser = build_argument_parser()
        required = ["--host", "127.0.0.1", "--port", "5556", "--frequency", "1"]
        args = parser.parse_args(
            required + ["--show-esi", "--show-telemetry"]
        )
        self.assertTrue(args.show_esi)
        self.assertEqual(args.show_telemetry, 30.0)

    def test_telemetry_interval_units(self):
        self.assertEqual(telemetry_interval("45"), 45.0)
        self.assertEqual(telemetry_interval("45s"), 45.0)
        self.assertEqual(telemetry_interval("5m"), 300.0)
        self.assertEqual(telemetry_interval("2h"), 7200.0)
        self.assertEqual(telemetry_interval("1.5m"), 90.0)

    def test_telemetry_interval_rejects_invalid_values(self):
        for value in ("0", "-1", "5d", "seconds", ""):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    telemetry_interval(value)

    def test_output_arguments_accept_no_value_or_directory(self):
        parser = build_argument_parser()
        required = ["--host", "127.0.0.1", "--port", "5556", "--frequency", "1"]
        self.assertEqual(
            parser.parse_args(required + ["--bits-output"]).bits_output,
            str(DEFAULT_OUTPUT_DIRECTORY),
        )
        self.assertEqual(
            parser.parse_args(
                required + ["--bits-output", "C:\\captures"]
            ).bits_output,
            "C:\\captures",
        )
        self.assertEqual(
            parser.parse_args(required + ["--iq-output"]).iq_output,
            str(DEFAULT_OUTPUT_DIRECTORY),
        )

    def test_log_arguments_accept_no_value_or_directory(self):
        parser = build_argument_parser()
        required = ["--host", "127.0.0.1", "--port", "5556", "--frequency", "1"]
        self.assertEqual(
            parser.parse_args(required + ["--log"]).log,
            str(DEFAULT_LOG_DIRECTORY),
        )
        self.assertEqual(
            parser.parse_args(required + ["--log", "C:\\logs"]).log,
            "C:\\logs",
        )
        self.assertEqual(
            parser.parse_args(required + ["--logdebug"]).logdebug,
            str(DEFAULT_LOG_DIRECTORY),
        )

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(required + ["--log", "--logdebug"])


if __name__ == "__main__":
    unittest.main()
