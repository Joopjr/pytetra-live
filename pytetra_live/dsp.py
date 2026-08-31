"""Stateful DSP and burst framing for live TETRA pi/4-DQPSK."""

from dataclasses import dataclass
import logging
import time

import numpy as np
from scipy import signal


LOG = logging.getLogger(__name__)
WORK_RATE = 72000.0
SYMBOL_RATE = 18000.0
SAMPLES_PER_SYMBOL = WORK_RATE / SYMBOL_RATE
BURST_BITS = 510
CHANNEL_CUTOFF = 16000.0

F_BITS = np.asarray([1] * 8 + [0] * 64 + [1] * 8, dtype=np.uint8)
N_BITS = np.asarray(
    [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    dtype=np.uint8,
)
P_BITS = np.asarray(
    [0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
    dtype=np.uint8,
)
Q_BITS = np.asarray(
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    dtype=np.uint8,
)
Y_BITS = np.asarray(
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0,
     0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    dtype=np.uint8,
)


def hamming(left, right):
    if len(left) != len(right):
        return max(len(left), len(right))
    return int(np.count_nonzero(np.asarray(left) != np.asarray(right)))


def rrc_taps(samples_per_symbol=4, alpha=0.35, span_symbols=10):
    count = span_symbols * samples_per_symbol
    times = np.arange(-count / 2, count / 2 + 1, dtype=np.float64)
    times /= float(samples_per_symbol)
    taps = np.empty_like(times)
    for index, value in enumerate(times):
        if abs(value) < 1e-12:
            taps[index] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif abs(abs(4.0 * alpha * value) - 1.0) < 1e-8:
            taps[index] = (
                alpha / np.sqrt(2.0)
                * ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                   + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha)))
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1.0 - alpha))
                + 4.0 * alpha * value * np.cos(np.pi * value * (1.0 + alpha))
            )
            denominator = np.pi * value * (1.0 - (4.0 * alpha * value) ** 2)
            taps[index] = numerator / denominator
    taps /= np.sqrt(np.sum(taps * taps))
    return taps.astype(np.float64)


def fourth_power_frequency_measurement(iq, sample_rate):
    if len(iq) < 4096:
        return 0.0, 0.0
    # For pi/4-DQPSK, the fourth power of every symbol-to-symbol
    # differential is -1.  A four-sample lag at the 4 SPS work rate removes
    # the data modulation while retaining four times the carrier phase.
    lag = int(round(sample_rate / SYMBOL_RATE))
    if len(iq) <= lag:
        return 0.0, 0.0
    differential = iq[lag:] * np.conj(iq[:-lag])
    differential /= np.maximum(np.abs(differential), 1e-12)
    coherent = -np.mean(differential ** 4)
    frequency = float(
        np.angle(coherent) * sample_rate / (2.0 * np.pi * 4.0 * lag)
    )
    return frequency, float(abs(coherent))


def fourth_power_frequency_estimate(iq, sample_rate):
    return fourth_power_frequency_measurement(iq, sample_rate)[0]


def robust_frequency_estimate(iq, sample_rate):
    """Combine coherent short-window estimates and reject weak outliers."""
    iq = np.asarray(iq, dtype=np.complex64)
    window = min(len(iq), max(8192, int(sample_rate * 0.25)))
    step = max(4096, window // 2)
    measurements = []
    for start in range(0, max(1, len(iq) - window + 1), step):
        frequency, coherence = fourth_power_frequency_measurement(
            iq[start:start + window], sample_rate
        )
        if np.isfinite(frequency) and coherence >= 0.02:
            measurements.append((frequency, coherence))
    if not measurements:
        return fourth_power_frequency_estimate(iq, sample_rate)
    frequencies = np.asarray([item[0] for item in measurements])
    weights = np.asarray([item[1] for item in measurements])
    median = float(np.median(frequencies))
    deviations = np.abs(frequencies - median)
    mad = float(np.median(deviations))
    tolerance = max(75.0, 4.0 * mad)
    keep = deviations <= tolerance
    return float(np.average(frequencies[keep], weights=weights[keep]))


class StreamingResampler:
    """Stateful linear complex resampler without chunk-boundary duplication."""

    def __init__(self, input_rate, output_rate=WORK_RATE):
        self.input_rate = float(input_rate)
        self.output_rate = float(output_rate)
        self.step = self.input_rate / self.output_rate
        self.tail = np.empty(0, dtype=np.complex64)
        self.position = 0.0

    def process(self, samples):
        samples = np.asarray(samples, dtype=np.complex64)
        data = np.concatenate((self.tail, samples))
        if len(data) < 2:
            self.tail = data
            return np.empty(0, dtype=np.complex64)
        positions = np.arange(self.position, len(data) - 1, self.step)
        indices = positions.astype(np.int64)
        fractions = positions - indices
        output = data[indices] + (data[indices + 1] - data[indices]) * fractions
        next_position = self.position + len(positions) * self.step
        self.tail = data[-1:].copy()
        self.position = next_position - (len(data) - 1)
        return output.astype(np.complex64, copy=False)


class FrequencyShifter:
    def __init__(self, frequency, sample_rate):
        self.frequency = float(frequency)
        self.sample_rate = float(sample_rate)
        self.phase = 0.0

    def process(self, samples):
        if not len(samples) or abs(self.frequency) < 1e-12:
            return samples
        step = 2.0 * np.pi * self.frequency / self.sample_rate
        phases = self.phase + step * np.arange(len(samples), dtype=np.float64)
        self.phase = float((phases[-1] + step) % (2.0 * np.pi))
        return (samples * np.exp(-1j * phases)).astype(np.complex64, copy=False)


class StreamingRmsNormalizer:
    """Slow level normalization for stable loops without pretending to add SNR."""

    def __init__(self, target_rms=0.5, time_constant=0.5):
        self.target_rms = float(target_rms)
        self.time_constant = float(time_constant)
        self.power = None

    def process(self, samples, sample_rate):
        if not len(samples):
            return samples
        block_power = float(np.mean(samples.real ** 2 + samples.imag ** 2))
        if not np.isfinite(block_power) or block_power <= 1e-15:
            return samples
        if self.power is None:
            self.power = block_power
        else:
            duration = len(samples) / float(sample_rate)
            alpha = 1.0 - np.exp(-duration / self.time_constant)
            self.power += alpha * (block_power - self.power)
        gain = self.target_rms / np.sqrt(max(self.power, 1e-15))
        gain = min(100.0, max(0.01, gain))
        return (samples * gain).astype(np.complex64, copy=False)


class CarrierPLL:
    def __init__(self, sample_rate, initial_frequency=0.0):
        self.sample_rate = float(sample_rate)
        self.frequency = float(initial_frequency)
        self.reference_frequency = float(initial_frequency)
        self.phase = 0.0
        self.coherence = 0.0
        self.last_measurement = float(initial_frequency)

    def process(self, samples, locked=False):
        if len(samples) >= 4096:
            estimate, coherence = fourth_power_frequency_measurement(
                samples, self.sample_rate
            )
            self.coherence += 0.10 * (coherence - self.coherence)
            self.last_measurement = float(estimate)
            # The estimator measures the complete current residual. Smooth it
            # slowly so a single faded block cannot retune the stream.
            if (
                np.isfinite(estimate)
                and abs(estimate) <= 2250.0
                and coherence >= 0.02
                and (not locked or abs(estimate - self.frequency) <= 450.0)
            ):
                if locked:
                    alpha = 0.004 if coherence < 0.08 else 0.010
                else:
                    alpha = 0.015 if coherence < 0.08 else 0.035
                self.frequency += alpha * (
                    estimate - self.frequency
                )
        step = 2.0 * np.pi * self.frequency / self.sample_rate
        phases = self.phase + step * np.arange(len(samples), dtype=np.float64)
        if len(phases):
            self.phase = float((phases[-1] + step) % (2.0 * np.pi))
        return (samples * np.exp(-1j * phases)).astype(np.complex64, copy=False)


class StreamingGardner:
    """Second-order Gardner recovery retaining phase and clock across blocks."""

    def __init__(self, phase=0.0, omega=SAMPLES_PER_SYMBOL):
        self.omega_nominal = SAMPLES_PER_SYMBOL
        self.omega = float(omega)
        self.position = float(phase) % self.omega_nominal
        self.buffer = np.empty(0, dtype=np.complex64)
        self.previous_symbol = None
        self.previous_position = None
        self.filtered_error = 0.0
        self.clock_error = 0.0
        self.energy_reference = 1e-6
        self.error_square_sum = 0.0
        self.error_count = 0

    @staticmethod
    def _interp(data, position):
        index = int(position)
        fraction = position - index
        return data[index] + (data[index + 1] - data[index]) * fraction

    def process(self, samples, locked=False):
        self.buffer = np.concatenate((self.buffer, np.asarray(samples, dtype=np.complex64)))
        if len(self.buffer) < 8:
            return np.empty(0, dtype=np.complex64)
        block_energy = float(np.percentile(np.abs(self.buffer) ** 2, 60))
        self.energy_reference += 0.05 * (max(block_energy, 1e-12) - self.energy_reference)
        data = self.buffer
        position = self.position
        omega = self.omega
        previous_symbol = self.previous_symbol
        previous_position = self.previous_position
        filtered_error = self.filtered_error
        clock_error = self.clock_error
        while position < 1.0:
            position += self.omega_nominal
        # Keep the feedback loop scalar, but avoid Python list growth, method
        # calls and repeated attribute lookups in this real-time hot path.
        capacity = max(1, int((len(data) - position) / 3.8) + 1)
        output = np.empty(capacity, dtype=np.complex64)
        output_count = 0
        while position + 1.0 < len(data) - 2:
            index = int(position)
            fraction = position - index
            current = data[index] + (data[index + 1] - data[index]) * fraction
            correction = 0.0
            if previous_symbol is not None:
                midpoint_position = 0.5 * (previous_position + position)
                midpoint_index = int(midpoint_position)
                midpoint_fraction = midpoint_position - midpoint_index
                midpoint = data[midpoint_index] + (
                    data[midpoint_index + 1] - data[midpoint_index]
                ) * midpoint_fraction
                energy = (
                    previous_symbol.real ** 2 + previous_symbol.imag ** 2
                    + midpoint.real ** 2 + midpoint.imag ** 2
                    + current.real ** 2 + current.imag ** 2
                )
                if energy > 0.15 * self.energy_reference:
                    error = float(
                        (np.conj(midpoint) * (previous_symbol - current)).real
                        / (energy + 1e-12)
                    )
                    error = max(-1.0, min(1.0, error))
                    self.error_square_sum += error * error
                    self.error_count += 1
                    filter_alpha = 0.025 if locked else 0.050
                    filtered_error += filter_alpha * (error - filtered_error)
                    clock_alpha = 0.0010 if locked else 0.0025
                    clock_error += clock_alpha * (filtered_error - clock_error)
                    stress = min(1.0, abs(clock_error) / 0.08)
                    omega_gain = (0.000025 + 0.000045 * stress) if locked else 0.00010
                    phase_gain = (0.008 + 0.008 * stress) if locked else 0.024
                    omega = max(3.8, min(4.2, omega + omega_gain * clock_error))
                    limit = 0.045 if locked else 0.09
                    correction = max(-limit, min(limit, phase_gain * filtered_error))
            output[output_count] = current
            output_count += 1
            previous_symbol = current
            previous_position = position
            position += omega + correction

        self.position = position
        self.omega = omega
        self.previous_symbol = previous_symbol
        self.previous_position = previous_position
        self.filtered_error = filtered_error
        self.clock_error = clock_error

        if self.previous_position is not None:
            keep_from = max(0, int(self.previous_position) - 2)
        else:
            keep_from = max(0, int(self.position) - 2)
        if keep_from:
            self.buffer = self.buffer[keep_from:]
            self.position -= keep_from
            if self.previous_position is not None:
                self.previous_position -= keep_from
        return output[:output_count]

    @property
    def error_rms(self):
        if not self.error_count:
            return 0.0
        return float(np.sqrt(self.error_square_sum / self.error_count))


def quadrants(symbols, previous_symbol=None, return_distances=False):
    if previous_symbol is not None:
        symbols = np.concatenate(([previous_symbol], symbols))
    if len(symbols) < 2:
        empty = np.empty(0, dtype=np.uint8)
        if return_distances:
            return empty, np.empty((0, 4), dtype=np.float32), previous_symbol
        return empty, previous_symbol
    normalized = symbols / np.maximum(np.abs(symbols), 1e-12)
    phase = np.angle(normalized[1:] * np.conj(normalized[:-1]))
    ideal = np.asarray([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    distance = np.abs(np.angle(np.exp(1j * (phase[:, None] - ideal[None, :]))))
    values = np.argmin(distance, axis=1).astype(np.uint8)
    if return_distances:
        return values, distance.astype(np.float32), symbols[-1]
    return values, symbols[-1]


def map_quadrants(values, inverse=False, bit_invert=False):
    mapping = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    if inverse:
        values = (-values) % 4
    bits = mapping[values].reshape(-1)
    if bit_invert:
        bits = 1 - bits
    return bits


def map_soft_quadrants(values, distances, inverse=False, bit_invert=False):
    """Return hard dibits and signed confidence from symbol distances.

    Positive confidence favours bit one, negative confidence favours zero.
    Magnitude expresses separation between the best competing hypotheses.
    """
    values = np.asarray(values, dtype=np.uint8)
    distances = np.asarray(distances, dtype=np.float32)
    bits = map_quadrants(values, inverse, bit_invert)
    if distances.shape != (len(values), 4):
        return bits, np.where(bits, 1.0, -1.0).astype(np.float32)

    labels = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    transformed = np.empty_like(labels)
    for original in range(4):
        mapped = (-original) % 4 if inverse else original
        transformed[original] = labels[mapped]
    if bit_invert:
        transformed = 1 - transformed

    confidence = np.empty((len(values), 2), dtype=np.float32)
    for bit_index in range(2):
        zero = np.min(distances[:, transformed[:, bit_index] == 0], axis=1)
        one = np.min(distances[:, transformed[:, bit_index] == 1], axis=1)
        confidence[:, bit_index] = zero - one
    scale = np.maximum(np.max(np.abs(confidence), axis=1, keepdims=True), 1e-6)
    confidence = np.clip(confidence / scale, -1.0, 1.0)
    return bits, confidence.reshape(-1)


def burst_quality(bits):
    if len(bits) != BURST_BITS:
        return None
    edge_errors = hamming(np.concatenate((bits[500:510], bits[0:12])), Q_BITS)
    sync_errors = hamming(bits[214:252], Y_BITS)
    frequency_errors = hamming(bits[14:94], F_BITS)
    normal_errors = min(hamming(bits[244:266], N_BITS), hamming(bits[244:266], P_BITS))
    if sync_errors <= 8 and frequency_errors <= 16 and edge_errors <= 6:
        return ("synchronization", sync_errors + frequency_errors + edge_errors)
    if edge_errors <= 2 and normal_errors <= 2:
        return ("normal", edge_errors + normal_errors)
    return None


class BurstFramer:
    """Resolve differential ambiguity and emit only structurally valid bursts."""

    variants = ((False, False), (False, True), (True, False), (True, True))

    def __init__(self, rejection_limit=12, acquisition_confidence=0.20):
        self.quadrant_buffer = np.empty(0, dtype=np.uint8)
        self.bits = np.empty(0, dtype=np.uint8)
        self.mapping = None
        self.locked = False
        self.rejected = 0
        self.consecutive_rejections = 0
        self.rejection_limit = max(1, int(rejection_limit))
        self.acquisition_confidence = float(acquisition_confidence)
        self.distance_buffer = np.empty((0, 4), dtype=np.float32)
        self.soft_bits = np.empty(0, dtype=np.float32)
        self.last_confidences = []
        self.accepted = 0
        self.training_error_sum = 0

    @staticmethod
    def _find_sync_candidates(bits):
        if len(bits) < BURST_BITS:
            return np.empty(0, dtype=np.int64)
        starts = np.arange(len(bits) - BURST_BITS + 1, dtype=np.int64)
        y_indices = starts[:, None] + 214 + np.arange(len(Y_BITS))
        y_errors = np.count_nonzero(bits[y_indices] != Y_BITS, axis=1)
        candidates = starts[y_errors <= 8]
        if not len(candidates):
            return np.empty(0, dtype=np.int64)
        f_indices = candidates[:, None] + 14 + np.arange(len(F_BITS))
        f_errors = np.count_nonzero(bits[f_indices] != F_BITS, axis=1)
        candidates = candidates[f_errors <= 16]
        if not len(candidates):
            return np.empty(0, dtype=np.int64)
        q_offsets = np.concatenate((np.arange(500, 510), np.arange(0, 12)))
        q_indices = candidates[:, None] + q_offsets
        q_errors = np.count_nonzero(bits[q_indices] != Q_BITS, axis=1)
        candidates = candidates[q_errors <= 6]
        return candidates

    def _find_confirmed_sync(self, bits, confidence):
        """Require a sync burst plus one valid burst at the exact next boundary."""
        training_offsets = np.concatenate((
            14 + np.arange(len(F_BITS)),
            214 + np.arange(len(Y_BITS)),
            np.arange(12),
            500 + np.arange(10),
        ))
        for candidate in self._find_sync_candidates(bits):
            start = int(candidate)
            if start + 2 * BURST_BITS > len(bits):
                continue
            if burst_quality(bits[start + BURST_BITS:start + 2 * BURST_BITS]) is None:
                continue
            training_confidence = float(
                np.mean(np.abs(confidence[start + training_offsets]))
            )
            if training_confidence >= self.acquisition_confidence:
                return start
        return None

    def feed(self, values, distances=None):
        values = np.asarray(values, dtype=np.uint8)
        if distances is None:
            distances = np.zeros((len(values), 4), dtype=np.float32)
            distances[np.arange(len(values)), values] = -1.0
        distances = np.asarray(distances, dtype=np.float32)
        bursts = []
        self.last_confidences = []
        gap = False
        if self.mapping is None:
            self.quadrant_buffer = np.concatenate((self.quadrant_buffer, values))
            self.distance_buffer = np.concatenate((self.distance_buffer, distances))
            for inverse, invert in self.variants:
                candidate, soft_candidate = map_soft_quadrants(
                    self.quadrant_buffer,
                    self.distance_buffer,
                    inverse,
                    invert,
                )
                start = self._find_confirmed_sync(candidate, soft_candidate)
                if start is not None:
                    self.mapping = (inverse, invert)
                    self.bits = candidate[start:]
                    self.soft_bits = soft_candidate[start:]
                    self.quadrant_buffer = np.empty(0, dtype=np.uint8)
                    self.distance_buffer = np.empty((0, 4), dtype=np.float32)
                    self.locked = True
                    self.consecutive_rejections = 0
                    LOG.info(
                        "TETRA burst lock acquired: inverse=%s bit_invert=%s",
                        inverse,
                        invert,
                    )
                    break
            if self.mapping is None:
                max_quadrants = 24 * 255
                if len(self.quadrant_buffer) > max_quadrants:
                    self.quadrant_buffer = self.quadrant_buffer[-max_quadrants:]
                    self.distance_buffer = self.distance_buffer[-max_quadrants:]
                return bursts, gap
        else:
            mapped, soft_mapped = map_soft_quadrants(
                values, distances, *self.mapping
            )
            self.bits = np.concatenate((self.bits, mapped))
            self.soft_bits = np.concatenate((self.soft_bits, soft_mapped))

        while len(self.bits) >= BURST_BITS:
            burst = self.bits[:BURST_BITS]
            quality = burst_quality(burst)
            if quality is None:
                self.rejected += 1
                self.consecutive_rejections += 1
                gap = True
                # Never invent replacement bits. Drop exactly the damaged
                # burst, notify downstream stateful decoders of the gap, and
                # preserve carrier/timing/mapping lock through short fades.
                self.bits = self.bits[BURST_BITS:]
                self.soft_bits = self.soft_bits[BURST_BITS:]
                if self.consecutive_rejections < self.rejection_limit:
                    continue

                self.mapping = None
                self.locked = False
                self.quadrant_buffer = np.empty(0, dtype=np.uint8)
                self.distance_buffer = np.empty((0, 4), dtype=np.float32)
                self.bits = np.empty(0, dtype=np.uint8)
                self.soft_bits = np.empty(0, dtype=np.float32)
                self.consecutive_rejections = 0
                LOG.info(
                    "TETRA burst lock released after %d consecutive rejected bursts",
                    self.rejection_limit,
                )
                break
            bursts.append(burst.copy())
            self.accepted += 1
            self.training_error_sum += int(quality[1])
            self.last_confidences.append(self.soft_bits[:BURST_BITS].copy())
            self.consecutive_rejections = 0
            self.bits = self.bits[BURST_BITS:]
            self.soft_bits = self.soft_bits[BURST_BITS:]
        return bursts, gap


@dataclass
class DemodulatorStats:
    input_samples: int = 0
    symbols: int = 0
    bursts: int = 0
    lock_losses: int = 0
    processing_seconds: float = 0.0
    resample_seconds: float = 0.0
    channel_filter_seconds: float = 0.0
    filter_seconds: float = 0.0
    carrier_seconds: float = 0.0
    timing_seconds: float = 0.0
    framing_seconds: float = 0.0
    input_level_dbfs: float = float("-inf")
    estimated_snr_db: float = float("nan")
    phase_error_rms_degrees: float = 0.0


class SignalQualityMonitor:
    """Low-rate RF measurements which never influence demodulation."""

    def __init__(self, sample_rate):
        self.sample_rate = float(sample_rate)
        self.input_power = None
        self.estimated_snr_db = float("nan")
        self.samples_since_spectrum = 0

    def update(self, samples):
        samples = np.asarray(samples, dtype=np.complex64)
        if not len(samples):
            return
        power = float(np.mean(np.abs(samples) ** 2))
        duration = len(samples) / self.sample_rate
        alpha = 1.0 - np.exp(-duration / 1.0)
        if np.isfinite(power) and power > 0.0:
            self.input_power = power if self.input_power is None else (
                self.input_power + alpha * (power - self.input_power)
            )
        self.samples_since_spectrum += len(samples)
        if self.samples_since_spectrum < self.sample_rate:
            return
        self.samples_since_spectrum = 0
        count = min(len(samples), 65536)
        if count < 4096:
            return
        window = np.hanning(count)
        spectrum = np.fft.fftshift(np.fft.fft(samples[-count:] * window))
        frequencies = np.fft.fftshift(np.fft.fftfreq(count, 1.0 / self.sample_rate))
        psd = np.abs(spectrum) ** 2 / max(float(np.sum(window ** 2)), 1.0)
        channel = np.abs(frequencies) <= 12500.0
        noise = (np.abs(frequencies) >= 18000.0) & (
            np.abs(frequencies) <= min(40000.0, 0.45 * self.sample_rate)
        )
        if not np.any(channel) or np.count_nonzero(noise) < 32:
            return
        noise_per_bin = float(np.median(psd[noise]))
        noise_in_channel = noise_per_bin * int(np.count_nonzero(channel))
        signal_power = max(float(np.sum(psd[channel])) - noise_in_channel, 1e-20)
        self.estimated_snr_db = float(
            10.0 * np.log10(signal_power / max(noise_in_channel, 1e-20))
        )

    @property
    def input_level_dbfs(self):
        if self.input_power is None:
            return float("-inf")
        return float(10.0 * np.log10(max(self.input_power, 1e-20)))


class LiveTetraDemodulator:
    def __init__(
        self,
        input_rate,
        channel_offset=0.0,
        acquisition_seconds=2.0,
        nominal_frequency=None,
    ):
        self.input_rate = float(input_rate)
        self.shifter = FrequencyShifter(float(channel_offset), self.input_rate)
        self.nominal_frequency = nominal_frequency
        channel_taps = min(
            257,
            max(65, int(round(self.input_rate / 1000.0)) | 1),
        )
        self.channel_taps = signal.firwin(
            channel_taps,
            CHANNEL_CUTOFF,
            fs=self.input_rate,
            window=("kaiser", 7.0),
        ).astype(np.float64)
        self.channel_filter_state = np.zeros(
            len(self.channel_taps) - 1, dtype=np.complex128
        )
        self.normalizer = StreamingRmsNormalizer()
        self.signal_monitor = SignalQualityMonitor(self.input_rate)
        self.resampler = StreamingResampler(self.input_rate, WORK_RATE)
        self.taps = rrc_taps()
        self.filter_state = np.zeros(len(self.taps) - 1, dtype=np.complex128)
        self.acquisition_samples = int(float(acquisition_seconds) * WORK_RATE)
        self.acquisition = []
        self.carrier = None
        self.timing = StreamingGardner()
        self.previous_symbol = None
        self.framer = BurstFramer()
        self.stats = DemodulatorStats()
        self.last_confidences = []

    def reset(self):
        self.__init__(
            self.input_rate,
            self.shifter.frequency,
            nominal_frequency=self.nominal_frequency,
        )

    def recover_stream(self):
        """Reset discontinuous stream state while retaining carrier lock."""
        retained_frequency = (
            self.carrier.frequency if self.carrier is not None else None
        )
        self.resampler = StreamingResampler(self.input_rate, WORK_RATE)
        self.channel_filter_state = np.zeros(
            len(self.channel_taps) - 1, dtype=np.complex128
        )
        self.normalizer = StreamingRmsNormalizer()
        self.signal_monitor = SignalQualityMonitor(self.input_rate)
        self.filter_state = np.zeros(len(self.taps) - 1, dtype=np.complex128)
        self.acquisition = []
        self.carrier = (
            CarrierPLL(WORK_RATE, retained_frequency)
            if retained_frequency is not None
            else None
        )
        self.timing = StreamingGardner()
        self.previous_symbol = None
        self.framer = BurstFramer()
        self.last_confidences = []
        return retained_frequency

    def process(self, iq):
        process_started = time.perf_counter()
        self.stats.input_samples += len(iq)
        shifted = self.shifter.process(np.asarray(iq, dtype=np.complex64))
        self.signal_monitor.update(shifted)
        self.stats.input_level_dbfs = self.signal_monitor.input_level_dbfs
        self.stats.estimated_snr_db = self.signal_monitor.estimated_snr_db
        stage_started = time.perf_counter()
        channel_filtered, self.channel_filter_state = signal.lfilter(
            self.channel_taps,
            [1.0],
            shifted,
            zi=self.channel_filter_state,
        )
        channel_filtered = self.normalizer.process(
            channel_filtered.astype(np.complex64, copy=False),
            self.input_rate,
        )
        self.stats.channel_filter_seconds += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        resampled = self.resampler.process(channel_filtered)
        self.stats.resample_seconds += time.perf_counter() - stage_started
        if not len(resampled):
            self.stats.processing_seconds += time.perf_counter() - process_started
            return [], False
        stage_started = time.perf_counter()
        filtered, self.filter_state = signal.lfilter(
            self.taps, [1.0], resampled, zi=self.filter_state
        )
        filtered = filtered.astype(np.complex64)
        self.stats.filter_seconds += time.perf_counter() - stage_started
        if self.carrier is None:
            self.acquisition.append(filtered)
            available = sum(len(block) for block in self.acquisition)
            if available < self.acquisition_samples:
                self.stats.processing_seconds += time.perf_counter() - process_started
                return [], False
            block = np.concatenate(self.acquisition)
            self.acquisition = []
            estimate = robust_frequency_estimate(block, WORK_RATE)
            if self.nominal_frequency is None:
                LOG.info("Initial residual carrier estimate: %+.2f Hz", estimate)
            else:
                LOG.info(
                    "Carrier acquisition: nominal=%d Hz residual=%+.2f Hz "
                    "effective=%.2f Hz search_limit=2250 Hz",
                    self.nominal_frequency,
                    estimate,
                    self.nominal_frequency + estimate,
                )
            self.carrier = CarrierPLL(WORK_RATE, estimate)
            filtered = block
        stage_started = time.perf_counter()
        corrected = self.carrier.process(filtered, locked=self.framer.locked)
        self.stats.carrier_seconds += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        symbols = self.timing.process(corrected, locked=self.framer.locked)
        self.stats.timing_seconds += time.perf_counter() - stage_started
        self.stats.symbols += len(symbols)
        values, distances, self.previous_symbol = quadrants(
            symbols, self.previous_symbol, return_distances=True
        )
        if len(values):
            nearest = distances[np.arange(len(values)), values]
            self.stats.phase_error_rms_degrees = float(
                np.degrees(np.sqrt(np.mean(nearest * nearest)))
            )
        stage_started = time.perf_counter()
        bursts, gap = self.framer.feed(values, distances)
        self.stats.framing_seconds += time.perf_counter() - stage_started
        self.last_confidences = self.framer.last_confidences
        self.stats.bursts += len(bursts)
        if gap:
            self.stats.lock_losses += 1
        self.stats.processing_seconds += time.perf_counter() - process_started
        return bursts, gap
