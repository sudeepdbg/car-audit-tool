"""Car Audit Tool - AI image comparison models package."""

from .comparator import (
    CarImageComparator,
    find_cached_weight_file,
    extract_candidate_frames,
    extract_representative_frame,
    load_image_array,
    pick_best_matching_frame,
    VIDEO_EXTENSIONS,
)
from .database import AuditDatabase
from .auth import AuthManager, seed_default_admin

__all__ = [
    "CarImageComparator",
    "AuditDatabase",
    "AuthManager",
    "seed_default_admin",
    "find_cached_weight_file",
    "extract_candidate_frames",
    "extract_representative_frame",
    "load_image_array",
    "pick_best_matching_frame",
    "VIDEO_EXTENSIONS",
]
__version__ = "2.4.0"
