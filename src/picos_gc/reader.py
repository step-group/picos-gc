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

        # Stream header layout:
        #   bytes 0-1  : 'RC' magic
        #   bytes 4-7  : header size (40 bytes)
        #   bytes 8-11 : number of data points (uint32 LE)
        #   bytes 12-15: total stream size
        #   byte  16+  : data as float64 LE, in µV
        try:
            n_points = struct.unpack("<I", raw[8:12])[0]
        except struct.error as exc:
            raise ValueError(f"Cannot parse point count in '{filepath}'") from exc

        offset = 16
        expected_bytes = n_points * 8
        if len(raw) < offset + expected_bytes:
            raise ValueError(
                f"Stream too short in '{filepath}': expected {offset + expected_bytes} bytes"
            )
        signal_uV = np.array(
            struct.unpack("<" + str(n_points) + "d", raw[offset : offset + expected_bytes])
        )

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
