"""Stateful DSP and burst framing for live TETRA pi/4-DQPSK."""

from dataclasses import dataclass
import logging

import numpy as np
from scipy import signal


LOG = logging.getLogger(__name__)
WORK_RATE = 72000.0
SYMBOL_RATE = 18000.0
SAMPLES_PER_SYMBOL = WORK_RATE / SYMBOL_RATE
BURST_BITS = 510

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


def fourth_power_frequency_estimate(iq, sample_rate):
    if len(iq) < 4096:
        return 0.0
    # For pi/4-DQPSK, the fourth power of every symbol-to-symbol
    # differential is -1.  A four-sample lag at the 4 SPS work rate removes
    # the data modulation while retaining four times the carrier phase.
    lag = int(round(sample_rate / SYMBOL_RATE))
    if len(iq) <= lag:
        return 0.0
    differential = iq[lag:] * np.conj(iq[:-lag])
    differential /= np.maximum(np.abs(differential), 1e-12)
    coherent = -np.mean(differential ** 4)
    return float(
        np.angle(coherent) * sample_rate / (2.0 * np.pi * 4.0 * lag)
    )


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


class CarrierPLL:
    def __init__(self, sample_rate, initial_frequency=0.0):
        self.sample_rate = float(sample_rate)
        self.frequency = float(initial_frequency)
        self.reference_frequency = float(initial_frequency)
        self.phase = 0.0
        self.frequency_alpha = 0.02

    def process(self, samples):
        if len(samples) >= 4096:
            estimate = fourth_power_frequency_estimate(samples, self.sample_rate)
            # The estimator measures the complete current residual. Smooth it
            # slowly so a single faded block cannot retune the stream.
            if np.isfinite(estimate) and abs(estimate) <= 2250.0:
                self.frequency += self.frequency_alpha * (
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

    @staticmethod
    def _interp(data, position):
        index = int(position)
        fraction = position - index
        return data[index] + (data[index + 1] - data[index]) * fraction

    def process(self, samples):
        self.buffer = np.concatenate((self.buffer, np.asarray(samples, dtype=np.complex64)))
        if len(self.buffer) < 8:
            return np.empty(0, dtype=np.complex64)
        block_energy = float(np.percentile(np.abs(self.buffer) ** 2, 60))
        self.energy_reference += 0.05 * (max(block_energy, 1e-12) - self.energy_reference)
        while self.position < 1.0:
            self.position += self.omega_nominal
        output = []
        while self.position + 1.0 < len(self.buffer) - 2:
            current = self._interp(self.buffer, self.position)
            correction = 0.0
            if self.previous_symbol is not None:
                midpoint_position = 0.5 * (self.previous_position + self.position)
                midpoint = self._interp(self.buffer, midpoint_position)
                energy = (
                    abs(self.previous_symbol) ** 2 + abs(midpoint) ** 2 + abs(current) ** 2
                )
                if energy > 0.15 * self.energy_reference:
                    error = float(
                        np.real(np.conj(midpoint) * (self.previous_symbol - current))
                        / (energy + 1e-12)
                    )
                    error = float(np.clip(error, -1.0, 1.0))
                    self.filtered_error += 0.04 * (error - self.filtered_error)
                    self.clock_error += 0.002 * (self.filtered_error - self.clock_error)
                    self.omega += 0.00008 * self.clock_error
                    self.omega = float(np.clip(self.omega, 3.8, 4.2))
                    correction = float(np.clip(0.020 * self.filtered_error, -0.08, 0.08))
            output.append(current)
            self.previous_symbol = current
            self.previous_position = self.position
            self.position += self.omega + correction

        if self.previous_position is not None:
            keep_from = max(0, int(self.previous_position) - 2)
        else:
            keep_from = max(0, int(self.position) - 2)
        if keep_from:
            self.buffer = self.buffer[keep_from:]
            self.position -= keep_from
            if self.previous_position is not None:
                self.previous_position -= keep_from
        return np.asarray(output, dtype=np.complex64)


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

    def __init__(self, rejection_limit=3):
        self.quadrant_buffer = np.empty(0, dtype=np.uint8)
        self.bits = np.empty(0, dtype=np.uint8)
        self.mapping = None
        self.locked = False
        self.rejected = 0
        self.consecutive_rejections = 0
        self.rejection_limit = max(1, int(rejection_limit))
        self.distance_buffer = np.empty((0, 4), dtype=np.float32)
        self.soft_bits = np.empty(0, dtype=np.float32)
        self.last_confidences = []

    @staticmethod
    def _find_sync(bits):
        if len(bits) < BURST_BITS:
            return None
        for start in range(0, len(bits) - BURST_BITS + 1):
            quality = burst_quality(bits[start:start + BURST_BITS])
            if quality is not None and quality[0] == "synchronization":
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
                start = self._find_sync(candidate)
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

    def process(self, iq):
        self.stats.input_samples += len(iq)
        shifted = self.shifter.process(np.asarray(iq, dtype=np.complex64))
        resampled = self.resampler.process(shifted)
        if not len(resampled):
            return [], False
        filtered, self.filter_state = signal.lfilter(
            self.taps, [1.0], resampled, zi=self.filter_state
        )
        filtered = filtered.astype(np.complex64)
        if self.carrier is None:
            self.acquisition.append(filtered)
            available = sum(len(block) for block in self.acquisition)
            if available < self.acquisition_samples:
                return [], False
            block = np.concatenate(self.acquisition)
            self.acquisition = []
            estimate = fourth_power_frequency_estimate(block, WORK_RATE)
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
        corrected = self.carrier.process(filtered)
        symbols = self.timing.process(corrected)
        self.stats.symbols += len(symbols)
        values, distances, self.previous_symbol = quadrants(
            symbols, self.previous_symbol, return_distances=True
        )
        bursts, gap = self.framer.feed(values, distances)
        self.last_confidences = self.framer.last_confidences
        self.stats.bursts += len(bursts)
        if gap:
            self.stats.lock_losses += 1
        return bursts, gap
