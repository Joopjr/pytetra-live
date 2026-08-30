# Testing with a real SpyServer

Automated tests validate protocol framing and DSP behavior, but the first RF
test must run on the network that can reach SpyServer.

## 1. Connection and IQ recording

Start with PyTetra delivery disabled:

```bash
pytetra-live \
    --host 127.0.0.1 \
    --port 5556 \
    --frequency 392475000 \
    --duration 15 \
    --no-decode \
    --iq-output first-test.cf32 \
    --bits-output first-test.bits \
    --debug
```

Replace the frequency if the active TETRA downlink differs. With no explicit
center frequency or gain, the current SpyServer values must be reported and
retained.

Verify that the log contains:

- device and client-sync reception;
- a negotiated sample rate;
- current center frequency and gain;
- initial residual carrier estimate;
- TETRA burst lock, if the selected channel contains a usable signal;
- final sample, burst, reconnect, and sequence-gap counts.

An empty bit file is valid when no structurally valid TETRA burst was found. It
must not be populated with guessed data.

## 2. Direct PyTetra decoding

After the sibling PyTetra project is installed:

```bash
pytetra-live \
    --host 127.0.0.1 \
    --port 5556 \
    --frequency 392475000 \
    --duration 30
```

Compact PyTetra MAC or Layer-3 records should appear after carrier acquisition
and synchronization. Add `--debug` only when complete per-layer diagnostics are
needed.

## 3. Explicit channel offset

When the TETRA channel is not at the current SpyServer IQ center:

```bash
pytetra-live \
    --host 127.0.0.1 \
    --port 5556 \
    --center-frequency 392500000 \
    --frequency 392475000 \
    --duration 30
```

The absolute offset plus 25 kHz safety margin must fit inside half the
negotiated IQ sample rate.

## Reporting a failed test

Retain:

- the complete debug log;
- `first-test.cf32`;
- `first-test.bits`;
- the exact command;
- SpyServer device type and configuration.

Do not publish captures or decoded subscriber information without appropriate
authorization.

