"""GCD binary file reader for Shimadzu chromatography data."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import olefile


@dataclass
class Chromatogram:
    filepath: Path
    time_min: np.ndarray  # shape (n,)
    signal_mV: np.ndarray  # shape (n,)


def read_gcd(filepath: Path | str) -> Chromatogram:
    """Read a Shimadzu .gcd file and return a Chromatogram.

    The .gcd format is an OLE2 Compound Document. Raw signal data lives in
    'LSS Raw Data/Chromatogram Ch1' as little-endian float64 values in µV.
    Total acquisition time is read from 'LSS Raw Data/Chromatogram Status'.

    Raises:
        ValueError: if the file cannot be parsed.
    """
    filepath = Path(filepath)
    try:
        ole = olefile.OleFileIO(filepath)
    except Exception as exc:
        raise ValueError(f"Cannot open '{filepath}' as OLE2 file: {exc}") from exc

    try:
        raw = ole.openstream("LSS Raw Data/Chromatogram Ch1").read()

        # Stream header layout (24-byte header, verified against real files):
        #   bytes 0-1  : 'RC' magic
        #   bytes 8-11 : number of data points (uint32 LE)
        #   bytes 12-15: total stream size in bytes (uint32 LE)
        #   bytes 16-23: zero padding
        #   bytes 24+  : signal as float64 LE, in µV (one value per data point)
        # NB: the data begins at byte 24, not 16 — reading from 16 prepends the
        # 8 zero-padding bytes as a spurious leading sample and drops the last.
        try:
            n_points = struct.unpack("<I", raw[8:12])[0]
        except struct.error as exc:
            raise ValueError(f"Cannot parse point count in '{filepath}'") from exc

        offset = 24
        expected_bytes = n_points * 8
        if len(raw) < offset + expected_bytes:
            raise ValueError(
                f"Stream too short in '{filepath}': expected at least "
                f"{offset + expected_bytes} bytes, got {len(raw)}"
            )
        signal_uV = np.frombuffer(raw, dtype="<f8", count=n_points, offset=offset)

        status = ole.openstream("LSS Raw Data/Chromatogram Status").read()
        try:
            total_ms = struct.unpack("<I", status[8:12])[0]
        except struct.error as exc:
            raise ValueError(f"Cannot parse total time in '{filepath}'") from exc

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Error reading streams from '{filepath}': {exc}") from exc
    finally:
        ole.close()

    time_min = np.linspace(0, total_ms / 60000.0, n_points)
    signal_mV = signal_uV / 1000.0

    return Chromatogram(filepath=filepath, time_min=time_min, signal_mV=signal_mV)


def time_to_index(chrom: Chromatogram, t: float) -> int:
    """Map a retention time (min) to the nearest sample index, clamped in range.

    Assumes the uniform sampling produced by :func:`read_gcd`.
    """
    n = len(chrom.time_min)
    if n == 0:
        return 0
    span = float(chrom.time_min[-1] - chrom.time_min[0])
    if span <= 0:
        return 0
    idx = round((t - float(chrom.time_min[0])) / span * (n - 1))
    return max(0, min(idx, n - 1))
