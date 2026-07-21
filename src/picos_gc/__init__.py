"""picos-gc: automatic multi-peak integration for Shimadzu .gcd files."""

from .aligner import AlignmentResult, Compound, align_peaks, save_aligned_csv
from .detector import DetectedPeak, DetectionParams, detect_peaks, estimate_noise
from .integrator import PeakResult, integrate_all_peaks, integrate_peak
from .processor import FileResult, process_batch, process_file, save_csv
from .reader import Chromatogram, read_gcd, time_to_index

__all__ = [
    "AlignmentResult",
    "Chromatogram",
    "Compound",
    "DetectedPeak",
    "DetectionParams",
    "FileResult",
    "PeakResult",
    "align_peaks",
    "detect_peaks",
    "estimate_noise",
    "integrate_all_peaks",
    "integrate_peak",
    "process_batch",
    "process_file",
    "read_gcd",
    "save_aligned_csv",
    "save_csv",
    "time_to_index",
]
