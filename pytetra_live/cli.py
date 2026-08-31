"""Command-line interface for PyTetra-live."""

import argparse
import logging
import signal

from . import __version__
from .bridge import PyTetraUnavailable
from .receiver import LiveReceiver


def positive_port(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def positive_frequency(value):
    frequency = int(value)
    if frequency <= 0:
        raise argparse.ArgumentTypeError("frequency must be positive")
    return frequency


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Receive SpyServer IQ, demodulate TETRA, and feed PyTetra"
    )
    parser.add_argument("--host", required=True, help="SpyServer host or IP address")
    parser.add_argument("--port", required=True, type=positive_port, help="SpyServer TCP port")
    parser.add_argument(
        "--frequency",
        required=True,
        type=positive_frequency,
        help="TETRA downlink channel frequency in Hz",
    )
    parser.add_argument(
        "--center-frequency",
        type=positive_frequency,
        help=(
            "optional per-client IQ center frequency; otherwise center the "
            "stream on --frequency"
        ),
    )
    parser.add_argument("--gain", type=int, help="optional device gain index")
    parser.add_argument("--sample-rate", type=float, help="minimum requested IQ sample rate")
    parser.add_argument("--bits-output", help="append validated bursts to an unpacked .bits file")
    parser.add_argument("--iq-output", help="append normalized interleaved float32 IQ")
    parser.add_argument("--no-decode", action="store_true", help="do not feed bursts into PyTetra")
    parser.add_argument("--no-reconnect", action="store_true", help="stop after a connection failure")
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--duration", type=float, help="optional run duration in seconds")
    parser.add_argument("--debug", action="store_true", help="enable DSP and complete PyTetra diagnostics")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    for name in ("sample_rate", "reconnect_delay", "timeout", "duration"):
        value = getattr(arguments, name)
        if value is not None and value <= 0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    logging.basicConfig(
        level=logging.DEBUG if arguments.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        receiver = LiveReceiver(
            host=arguments.host,
            port=arguments.port,
            frequency=arguments.frequency,
            center_frequency=arguments.center_frequency,
            gain=arguments.gain,
            sample_rate=arguments.sample_rate,
            debug=arguments.debug,
            decode=not arguments.no_decode,
            bits_output=arguments.bits_output,
            iq_output=arguments.iq_output,
            reconnect=not arguments.no_reconnect,
            reconnect_delay=arguments.reconnect_delay,
            timeout=arguments.timeout,
        )
    except PyTetraUnavailable as exc:
        parser.error(str(exc))

    def request_stop(unused_signum, unused_frame):
        receiver.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        stats = receiver.run(duration=arguments.duration)
    except (ConnectionError, OSError, ValueError, RuntimeError) as exc:
        logging.error("PyTetra-live stopped: %s", exc)
        return 1
    logging.info(
        "Stopped: connections=%d reconnects=%d samples=%d bursts=%d "
        "sequence_gaps=%d queue_overruns=%d decoder_overruns=%d",
        stats.connections,
        stats.reconnects,
        stats.iq_samples,
        stats.bursts,
        stats.sequence_gaps,
        stats.queue_overruns,
        stats.decoder_overruns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
