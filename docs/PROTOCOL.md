# SpyServer client protocol notes

PyTetra-live implements the little-endian SpyServer 2.0.1700 client structures
needed for IQ streaming. Definitions were checked against the public protocol
header used by the SDR++ SpyServer source module.

## Connection sequence

1. Open TCP connection.
2. Send HELLO with protocol version and application name.
3. Receive device information and client synchronization.
4. Select supported IQ format and a safe decimation stage.
5. Request a per-client IQ center frequency and retain or explicitly request gain.
6. Select IQ-only mode.
7. Enable streaming.
8. Parse framed messages and verify sequence continuity.
9. Disable streaming and close the socket.

## Message safety

- command bodies are limited to 256 bytes;
- message bodies are limited to 1 MiB;
- all fixed structures are size-checked before unpacking;
- incomplete TCP reads are reassembled exactly;
- invalid IQ body alignment is rejected;
- INT24 IQ is rejected rather than misinterpreted;
- a sequence discontinuity is logged and counted.

## Frequency behavior

SpyServer reports both the shared device center and the permitted IQ-center
range. `SETTING_IQ_FREQUENCY` selects the digital center for this client and
does not require moving the shared tuner while the request remains inside that
range. PyTetra-live centers a default stream on the requested TETRA channel.
An explicit `--center-frequency` overrides that choice.

If a locked server cannot provide the requested IQ center, PyTetra-live retains
the synchronized IQ center, negotiates enough bandwidth, and performs the
remaining channel offset locally.
