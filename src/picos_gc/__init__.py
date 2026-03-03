"""picos-gc: automatic multi-peak integration for Shimadzu .gcd files."""

from .aligner import AlignmentResult, Compound, align_peaks, save_aligned_csv
from .detector import DetectedPeak, DetectionParams, detect_peaks
from .integrator import PeakResult, integrate_all_peaks, integrate_peak
from .processor import FileResult, process_batch, process_file, save_csv
from .reader import Chromatogram, read_gcd

__all__ = [
    "AlignmentResult",
    "Compound",
    "align_peaks",
    "save_aligned_csv",
    "Chromatogram",
    "read_gcd",
    "DetectedPeak",
    "DetectionParams",
    "detect_peaks",
    "PeakResult",
    "integrate_peak",
    "integrate_all_peaks",
    "FileResult",
    "process_file",
    "process_batch",
    "save_csv",
]
