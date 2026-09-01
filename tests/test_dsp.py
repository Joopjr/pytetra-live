import unittest
from pathlib import Path

import numpy as np
from scipy import signal

from pytetra_live.dsp import (
    BurstFramer,
    CarrierPLL,
    LiveTetraDemodulator,
    SignalQualityMonitor,
    StreamingGardner,
    StreamingResampler,
    WORK_RATE,
    burst_quality,
    map_soft_quadrants,
    rrc_taps,
)


class DspTestCase(unittest.TestCase):
    @staticmethod
    def fixture_path():
        return Path(__file__).parent / "data" / "tetra_downlink.bits"

    def test_streaming_resampler_has_no_chunk_duplicates(self):
        source = np.arange(1000, dtype=np.float32).astype(np.complex64)
        one = StreamingResampler(1000, 800).process(source)
        split_resampler = StreamingResampler(1000, 800)
        split = np.concatenate(
            [split_resampler.process(source[:333]), split_resampler.process(source[333:])]
        )
        np.testing.assert_allclose(split, one, atol=1e-5)

    def test_gardner_clock_correction_is_bounded(self):
        recovery = StreamingGardner()
        rng = np.random.default_rng(19)
        # Exercise many small blocks with detector noise. The recovered clock
        # must never perform the random walk seen in the strong-signal log.
        for unused in range(300):
            samples = (
                rng.standard_normal(512) + 1j * rng.standard_normal(512)
            ).astype(np.complex64)
            recovery.process(samples, locked=True)

        drift_ppm = 1e6 * (
            recovery.omega / recovery.omega_nominal - 1.0
        )
        self.assertLessEqual(abs(drift_ppm), recovery.MAX_CLOCK_PPM + 1e-6)

    def test_demodulator_filter_pipeline_stays_single_precision(self):
        demodulator = LiveTetraDemodulator(93750.0)
        self.assertEqual(demodulator.channel_taps.dtype, np.float32)
        self.assertEqual(demodulator.channel_filter_state.dtype, np.complex64)
        self.assertEqual(demodulator.taps.dtype, np.float32)
        self.assertEqual(demodulator.filter_state.dtype, np.complex64)

        samples = np.zeros(2048, dtype=np.complex64)
        filtered, state = signal.lfilter(
            demodulator.channel_taps,
            demodulator.filter_denominator,
            samples,
            zi=demodulator.channel_filter_state,
        )
        self.assertEqual(filtered.dtype, np.complex64)
        self.assertEqual(state.dtype, np.complex64)

    def test_signal_monitor_reports_level_and_in_band_snr(self):
        sample_rate = 93750.0
        rng = np.random.default_rng(7)
        count = int(sample_rate)
        positions = np.arange(count, dtype=np.float64)
        noise = 0.02 * (
            rng.standard_normal(count) + 1j * rng.standard_normal(count)
        )
        tone = 0.20 * np.exp(1j * 2.0 * np.pi * 4000.0 * positions / sample_rate)
        monitor = SignalQualityMonitor(sample_rate)

        monitor.update((tone + noise).astype(np.complex64))

        self.assertGreater(monitor.input_level_dbfs, -20.0)
        self.assertGreater(monitor.estimated_snr_db, 10.0)

    def test_signal_monitor_accumulates_small_input_blocks(self):
        sample_rate = 93750.0
        positions = np.arange(int(sample_rate), dtype=np.float64)
        samples = (0.2 * np.exp(
            1j * 2.0 * np.pi * 4000.0 * positions / sample_rate
        )).astype(np.complex64)
        monitor = SignalQualityMonitor(sample_rate)

        for start in range(0, len(samples), 1024):
            monitor.update(samples[start:start + 1024])

        self.assertTrue(np.isfinite(monitor.estimated_snr_db))

    def test_locked_carrier_loop_rejects_large_single_jump(self):
        pll = CarrierPLL(WORK_RATE, 250.0)
        # Force a deterministic estimator result without coupling this test to
        # the waveform estimator's window behaviour.
        original = __import__("pytetra_live.dsp", fromlist=["dummy"])
        measurement = original.fourth_power_frequency_measurement
        try:
            original.fourth_power_frequency_measurement = lambda samples, rate: (900.0, 0.5)
            pll.process(np.ones(2048, dtype=np.complex64), locked=True)
            pll.process(np.ones(6144, dtype=np.complex64), locked=True)
        finally:
            original.fourth_power_frequency_measurement = measurement

        self.assertEqual(pll.frequency, 250.0)

    def test_soft_confidence_polarity_matches_all_hard_mappings(self):
        values = np.asarray([0, 1, 2, 3], dtype=np.uint8)
        distances = np.full((4, 4), 2.0, dtype=np.float32)
        distances[np.arange(4), values] = 0.0
        for inverse, invert in BurstFramer.variants:
            bits, confidence = map_soft_quadrants(
                values, distances, inverse, invert
            )
            np.testing.assert_array_equal(confidence > 0.0, bits == 1)

    def test_public_example_contains_200_valid_bursts(self):
        path = self.fixture_path()
        bits = np.fromfile(path, dtype=np.uint8)
        qualities = [burst_quality(bits[i:i + 510]) for i in range(0, len(bits), 510)]
        self.assertEqual(len(qualities), 200)
        self.assertTrue(all(quality is not None for quality in qualities))

    def test_framer_resolves_mapping_and_emits_all_bursts(self):
        path = self.fixture_path()
        bits = np.fromfile(path, dtype=np.uint8).reshape(-1, 2)
        reverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        values = np.asarray([reverse[tuple(pair)] for pair in bits], dtype=np.uint8)
        framer = BurstFramer()
        bursts = []
        for start in range(0, len(values), 137):
            decoded, gap = framer.feed(values[start:start + 137])
            self.assertFalse(gap)
            bursts.extend(decoded)
        self.assertEqual(len(bursts), 200)
        self.assertTrue(framer.locked)

    def test_framer_does_not_lock_on_one_unconfirmed_sync_burst(self):
        raw = np.fromfile(self.fixture_path(), dtype=np.uint8)[:2 * 510].copy()
        self.assertEqual(burst_quality(raw[:510])[0], "synchronization")
        raw[510:] = 0
        reverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        values = np.asarray(
            [reverse[tuple(pair)] for pair in raw.reshape(-1, 2)],
            dtype=np.uint8,
        )
        framer = BurstFramer()

        bursts, gap = framer.feed(values)

        self.assertEqual(bursts, [])
        self.assertFalse(gap)
        self.assertFalse(framer.locked)

    def test_sync_candidates_must_start_on_a_dibit_boundary(self):
        raw = np.fromfile(self.fixture_path(), dtype=np.uint8)[:510]
        odd_aligned = np.concatenate((np.asarray([0], dtype=np.uint8), raw))

        candidates = BurstFramer._find_sync_candidates(odd_aligned)

        self.assertNotIn(1, candidates)

    def test_channel_filter_rejects_out_of_channel_interference(self):
        demodulator = LiveTetraDemodulator(93750.0)
        frequencies, response = signal.freqz(
            demodulator.channel_taps,
            worN=8192,
            fs=demodulator.input_rate,
        )
        passband = abs(response[np.argmin(abs(frequencies - 12000.0))])
        adjacent = abs(response[np.argmin(abs(frequencies - 30000.0))])

        self.assertGreater(passband, 0.90)
        self.assertLess(adjacent, 0.01)

    def test_framer_keeps_lock_across_one_damaged_burst(self):
        raw = np.fromfile(self.fixture_path(), dtype=np.uint8)[:5 * 510].copy()
        # Confirm with three valid bursts, then damage one complete burst while
        # leaving the following burst perfectly aligned.
        # validation, while leaving the following burst perfectly aligned.
        raw[3 * 510:4 * 510] = 0
        reverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        values = np.asarray(
            [reverse[tuple(pair)] for pair in raw.reshape(-1, 2)],
            dtype=np.uint8,
        )
        framer = BurstFramer(rejection_limit=3)

        bursts, gap = framer.feed(values)

        self.assertTrue(gap)
        self.assertEqual(len(bursts), 4)
        self.assertTrue(framer.locked)
        self.assertEqual(framer.rejected, 1)
        self.assertEqual(framer.consecutive_rejections, 0)

    def test_framer_releases_lock_after_sustained_damage(self):
        raw = np.fromfile(self.fixture_path(), dtype=np.uint8)[:6 * 510].copy()
        raw[3 * 510:] = 0
        reverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        values = np.asarray(
            [reverse[tuple(pair)] for pair in raw.reshape(-1, 2)],
            dtype=np.uint8,
        )
        framer = BurstFramer(rejection_limit=3)

        bursts, gap = framer.feed(values)

        self.assertTrue(gap)
        self.assertEqual(len(bursts), 3)
        self.assertFalse(framer.locked)
        self.assertIsNone(framer.mapping)

    def test_default_framer_survives_five_damaged_bursts(self):
        raw = np.fromfile(self.fixture_path(), dtype=np.uint8)[:9 * 510].copy()
        raw[3 * 510:8 * 510] = 0
        reverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        values = np.asarray(
            [reverse[tuple(pair)] for pair in raw.reshape(-1, 2)],
            dtype=np.uint8,
        )
        framer = BurstFramer()

        bursts, gap = framer.feed(values)

        self.assertTrue(gap)
        self.assertEqual(len(bursts), 4)
        self.assertTrue(framer.locked)

    def test_live_pipeline_acquires_frequency_timing_and_bursts(self):
        path = self.fixture_path()
        raw_bits = np.fromfile(path, dtype=np.uint8)[:40 * 510]
        reverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        values = np.asarray(
            [reverse[tuple(pair)] for pair in raw_bits.reshape(-1, 2)],
            dtype=np.uint8,
        )
        ideal = np.asarray(
            [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
        )
        symbols = np.exp(1j * np.cumsum(ideal[values])).astype(np.complex64)
        transmit = np.zeros(len(symbols) * 4, dtype=np.complex64)
        transmit[::4] = symbols
        transmit = signal.lfilter(rrc_taps(), [1.0], transmit).astype(np.complex64)
        positions = np.arange(len(transmit), dtype=np.float64)
        transmit *= np.exp(1j * 2.0 * np.pi * 250.0 * positions / WORK_RATE)

        demodulator = LiveTetraDemodulator(
            WORK_RATE, acquisition_seconds=0.1
        )
        bursts = []
        for start in range(0, len(transmit), 1024):
            decoded, gap = demodulator.process(transmit[start:start + 1024])
            self.assertFalse(gap)
            bursts.extend(decoded)

        self.assertGreaterEqual(len(bursts), 38)
        self.assertTrue(demodulator.framer.locked)
        self.assertAlmostEqual(demodulator.carrier.frequency, 250.0, delta=15.0)
        timing_drift_ppm = 1e6 * (
            demodulator.timing.omega / demodulator.timing.omega_nominal - 1.0
        )
        self.assertLess(abs(timing_drift_ppm), 50.0)

    def test_warm_recovery_retains_carrier_without_reacquisition(self):
        demodulator = LiveTetraDemodulator(WORK_RATE)
        demodulator.carrier = CarrierPLL(WORK_RATE, 637.5)
        demodulator.acquisition = [np.ones(32, dtype=np.complex64)]
        demodulator.previous_symbol = 1.0 + 0.0j
        demodulator.framer.mapping = (False, False)

        retained = demodulator.recover_stream()

        self.assertEqual(retained, 637.5)
        self.assertEqual(demodulator.carrier.frequency, 637.5)
        self.assertEqual(demodulator.acquisition, [])
        self.assertIsNone(demodulator.previous_symbol)
        self.assertIsNone(demodulator.framer.mapping)


if __name__ == "__main__":
    unittest.main()
