# Changelog

## 1.0.0

- decoupled SpyServer socket reception from DSP processing with a bounded
  real-time IQ queue;
- coalesced clustered SpyServer loss into controlled DSP discontinuities and
  rate-limited repeated gap warnings;
- added a queue-overrun statistic to distinguish local DSP backpressure from
  network or SpyServer packet loss;
- retained the stable carrier estimate during local queue-overrun recovery,
  avoiding repeated full carrier acquisition while resetting only stream-
  dependent filter, timing, differential, and burst-alignment state;
- vectorized synchronization-sequence acquisition and batched queued IQ blocks
  to remove repeated Python scanning and per-message DSP overhead;

- added a minimal reconnectable SpyServer protocol client;
- added required host, port, and TETRA channel-frequency arguments;
- added per-client SpyServer IQ centering without moving the shared hardware
  center, including wide-stream local-tuning fallback;
- retained server gain unless explicitly overridden;
- added safe decimation negotiation and UINT8/INT16/FLOAT IQ conversion;
- added continuous channel shifting, resampling, RRC filtering, carrier PLL,
  Gardner timing recovery, and differential π/4-DQPSK demodulation;
- added bounded residual-frequency acquisition around the nominal TETRA
  channel and explicit effective-frequency diagnostics;
- fixed false initial status-message gaps and bandwidth selection for locked
  SpyServer sessions;
- added training-sequence burst framing without fabricated replacement bits;
- preserved carrier, timing, and mapping lock across isolated damaged bursts,
  with full reacquisition after three consecutive structural failures;
- added live soft-bit confidence and soft-decision Viterbi correction while
  retaining CRC-gated delivery and hard-bit file compatibility;
- removed an expensive repeated in-buffer alignment scan that could starve
  SpyServer reception during sustained weak-signal damage;
- added direct PyTetra delivery plus optional IQ and bit recording;
- added simulated-server, DSP, CLI, and public-example regression tests.
- licensed PyTetra-live under GNU General Public License v3.0 only.
