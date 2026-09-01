import unittest
import contextlib
import io

from pytetra_live.cli import build_argument_parser


class CliTestCase(unittest.TestCase):
    def test_required_connection_arguments(self):
        parser = build_argument_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
        args = parser.parse_args(
            ["--host", "127.0.0.1", "--port", "5556", "--frequency", "392475000"]
        )
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 5556)
        self.assertEqual(args.frequency, 392475000)
        self.assertIsNone(args.gain)
        self.assertIsNone(args.center_frequency)
        self.assertFalse(args.show_esi)
        self.assertFalse(args.show_security_context)

    def test_optional_protocol_output_arguments(self):
        parser = build_argument_parser()
        required = ["--host", "127.0.0.1", "--port", "5556", "--frequency", "1"]
        args = parser.parse_args(
            required + ["--show-esi", "--show-security-context"]
        )
        self.assertTrue(args.show_esi)
        self.assertTrue(args.show_security_context)

    def test_log_arguments_accept_no_value_or_directory(self):
        parser = build_argument_parser()
        required = ["--host", "127.0.0.1", "--port", "5556", "--frequency", "1"]
        self.assertEqual(parser.parse_args(required + ["--log"]).log, ".")
        self.assertEqual(
            parser.parse_args(required + ["--log", "C:\\logs"]).log,
            "C:\\logs",
        )
        self.assertEqual(parser.parse_args(required + ["--logdebug"]).logdebug, ".")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(required + ["--log", "--logdebug"])


if __name__ == "__main__":
    unittest.main()
