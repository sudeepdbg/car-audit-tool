"""
Car Image Comparator - ResNet50-based deep feature comparison engine
====================================================================

Features:
- Uses the modern torchvision `weights=` API.
- Prefers locally-cached weights so the tool runs fully offline after
  `python scripts/download_model.py` has been executed once.
- Falls back to an untrained model (loudly logged) if no weights are
  available, so the UI never crashes.
- Feature extraction, pixel metrics, heatmap generation and report assembly
  share one alignment pass: a single `compare_images()` call loads each image
  exactly once, runs ORB/RANSAC alignment once, and computes the SSIM map once.
  The pixel metrics, the change mask and the heatmap all describe the same
  (aligned) pixels, so the yellow region boxes always line up with the
  red overlay.
- Heatmap output includes a side-by-side composite (golden | returned |
  overlay).
- Accepts images OR short videos. A representative (sharpest) frame is
  extracted from a video automatically, then run through the same pipeline.

NOTE: the trust score is a *similarity heuristic*, not a trained damage
detector. It is meant to triage image pairs for human review.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import cv2

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.color import rgb2gray
except ImportError:  # pragma: no cover - skimage is in requirements
    ssim = None
    rgb2gray = None


logger = logging.getLogger("car_audit_tool.comparator")


# -----------------------------------------------------------------------------
# Weight loading helpers
# -----------------------------------------------------------------------------

_WEIGHT_ENUM = ResNet50_Weights.IMAGENET1K_V2


def find_cached_weight_file() -> Optional[str]:
    """Look for a cached resnet50 weights file under TORCH_HOME.

    Public alias of the legacy ``_find_cached_weight_file`` so callers outside
    this module (e.g. the health endpoint) don't have to reach into a private
    function. The underscore-prefixed name is kept for backwards compatibility.
    """
    return _find_cached_weight_file()


def _find_cached_weight_file() -> Optional[str]:
    """Look for a cached resnet50 weights file under TORCH_HOME."""
    torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    hub_dir = os.path.join(torch_home, "hub", "checkpoints")
    if not os.path.isdir(hub_dir):
        return None
    # Common filename patterns torchvision uses for ResNet50 weights
    candidates = [
        "resnet50-11ad3fa6.pth",   # IMAGENET1K_V2
        "resnet50-0676ba61.pth",   # IMAGENET1K_V1
    ]
    for name in candidates:
        full = os.path.join(hub_dir, name)
        if os.path.isfile(full):
            return full
    # Fallback: any resnet50*.pth
    for name in sorted(os.listdir(hub_dir)):
        if name.lower().startswith("resnet50") and name.lower().endswith(".pth"):
            return os.path.join(hub_dir, name)
    return None


def _strip_head(model: nn.Module, device: torch.device) -> nn.Module:
    """Drop the classification layer (fc) and move the backbone to `device`."""
    backbone = nn.Sequential(*list(model.children())[:-1])
    return backbone.to(device).eval()


def _load_resnet50_feature_extractor(device: torch.device) -> Tuple[nn.Module, str, str]:
    """
    Load ResNet50 with the classification head removed.

    Order of attempts (so an offline machine never waits on a network timeout
    when a local copy exists):
      1. a cached .pth file under TORCH_HOME
      2. the torchvision weights registry (downloads if online)
      3. an untrained network (loudly logged; scores are unreliable)

    Returns (model, model_label, weight_source).
    """
    last_error: Optional[Exception] = None

    cached = _find_cached_weight_file()
    if cached:
        try:
            model = models.resnet50(weights=None)
            try:
                state_dict = torch.load(cached, map_location="cpu", weights_only=True)
            except TypeError:  # very old torch without weights_only
                state_dict = torch.load(cached, map_location="cpu")
            model.load_state_dict(state_dict)  # strict: fail loudly on mismatch
            logger.info("Loaded ResNet50 weights from cached file: %s", cached)
            return _strip_head(model, device), "ResNet50 (cached weights)", f"cached file: {cached}"
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning("Failed to load cached weights (%s): %s", cached, e)

    try:
        model = models.resnet50(weights=_WEIGHT_ENUM)
        logger.info("Loaded ResNet50 weights via torchvision registry.")
        return _strip_head(model, device), "ResNet50 (ImageNet1k_V2)", "torchvision weights registry"
    except Exception as e:  # noqa: BLE001
        last_error = e
        logger.warning("Could not load weights via registry: %s", e)

    logger.error(
        "No ResNet50 weights available. Run `python scripts/download_model.py` "
        "once while online. Falling back to an UNTRAINED model - results are "
        "unreliable. Last error: %s",
        last_error,
    )
    model = models.resnet50(weights=None)
    return _strip_head(model, device), "ResNet50 (UNTRAINED - weights missing)", "untrained fallback"


# -----------------------------------------------------------------------------
# Image / video loading helpers
# -----------------------------------------------------------------------------

VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "wmv", "3gp", "mpg", "mpeg"}

# A video frame is only trusted as "the same view" when at least this many
# ORB matches survive RANSAC; below it we fall back to the sharpest frame.
MIN_MATCH_INLIERS = 15
# Longest side kept for a decoded video frame (bounds memory for 4K clips; the
# comparison pipeline itself caps at CarImageComparator.MAX_SIDE anyway).
VIDEO_FRAME_MAX_SIDE = 1600


def _is_video_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ext in VIDEO_EXTENSIONS


def _load_still(path: str) -> np.ndarray:
    """Decode an image as RGB uint8, honouring EXIF orientation.

    Phone cameras usually store pixels in sensor orientation and record the
    real orientation in EXIF. Ignoring that tag made rotated phone photos
    compare sideways / upside-down (and disagree with the browser preview,
    which does apply it).
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("RGB"))


def _sample_indices(frame_count: int, n: int) -> List[int]:
    """Evenly spaced frame indices from the middle 80% of a clip."""
    lo, hi = int(frame_count * 0.1), int(frame_count * 0.9)
    if hi <= lo:
        lo, hi = 0, max(frame_count - 1, 0)
    if n <= 1:
        return [(lo + hi) // 2]
    idx = {min(max(lo + int(round((hi - lo) * i / (n - 1))), 0), frame_count - 1) for i in range(n)}
    return sorted(idx)


def _make_candidate(frame_bgr: np.ndarray, index: int, fps: float) -> Dict[str, Any]:
    h, w = frame_bgr.shape[:2]
    if max(h, w) > VIDEO_FRAME_MAX_SIDE:
        s = VIDEO_FRAME_MAX_SIDE / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    try:
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except cv2.error:
        sharp = 0.0
    return {
        "frame": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        "index": int(index),
        "time_sec": round(index / fps, 2) if fps and fps > 0 else None,
        "sharpness": sharp,
    }


def extract_candidate_frames(video_path: str, sample_count: int = 7) -> List[Dict[str, Any]]:
    """Decode up to ``sample_count`` evenly spaced candidate frames from a video.

    Each candidate is ``{"frame": RGB uint8, "index", "time_sec", "sharpness"}``.
    Seeks when the container reports a trustworthy frame count; otherwise
    (browser-recorded WebM, some MKV/WMV files report 0 or -1) falls back to a
    bounded sequential scan so we never silently use only the first frame.
    Returns ``[]`` if the video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps):
            fps = 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out: List[Dict[str, Any]] = []
        if frame_count > 1:
            for idx in _sample_indices(frame_count, sample_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    out.append(_make_candidate(frame, idx, fps))
    finally:
        cap.release()
    if out:
        return out

    # Unknown / wrong frame count: bounded sequential scan on a fresh handle.
    cap = cv2.VideoCapture(video_path)
    try:
        kept: List[Dict[str, Any]] = []
        step, i = 8, 0
        while i < 480:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if i % step == 0:
                kept.append(_make_candidate(frame, i, fps))
            i += 1
        if len(kept) > sample_count:
            pick = np.linspace(0, len(kept) - 1, sample_count).round().astype(int)
            kept = [kept[j] for j in sorted(set(pick.tolist()))]
        return kept
    finally:
        cap.release()


def _sharpest(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(candidates, key=lambda c: c["sharpness"])


def extract_representative_frame_info(video_path: str, sample_count: int = 3) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Sharpest of ``sample_count`` frames plus ``{"index", "time_sec"}`` info, or None."""
    cands = extract_candidate_frames(video_path, sample_count)
    if not cands:
        return None
    best = _sharpest(cands)
    return best["frame"], {"index": best["index"], "time_sec": best["time_sec"]}


def extract_representative_frame(video_path: str, sample_count: int = 3) -> Optional[np.ndarray]:
    """Return one representative RGB frame (the sharpest of a few samples from
    the middle 80% of the clip), or None if the video cannot be read."""
    res = extract_representative_frame_info(video_path, sample_count)
    return res[0] if res else None


def _gray_for_matching(rgb: np.ndarray, max_side: int = 640) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    s = max_side / max(g.shape[:2])
    if s < 1.0:
        g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return g


def pick_best_matching_frame(candidates: List[Dict[str, Any]], reference_rgb: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Choose the candidate frame that shows the same view as ``reference_rgb``.

    A video contains many viewpoints; picking the sharpest frame on its own can
    pick a different side of the car than the golden photo shows, and the
    difference would then be reported as damage. Instead, rank the reasonably
    sharp candidates by how many ORB matches survive RANSAC against the
    reference. Falls back to the sharpest frame when nothing matches.
    The returned dict carries an extra ``match_inliers`` key.
    """
    if not candidates:
        return None
    fallback = dict(_sharpest(candidates), match_inliers=0)
    if reference_rgb is None or len(candidates) == 1:
        return fallback
    try:
        orb = cv2.ORB_create(1000)
        ref_kp, ref_desc = orb.detectAndCompute(_gray_for_matching(reference_rgb), None)
        if ref_desc is None or len(ref_kp) < 12:
            return fallback
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        top_sharp = max(c["sharpness"] for c in candidates)
        pool = [c for c in candidates if c["sharpness"] >= 0.4 * top_sharp] or candidates
        best, best_score = None, 0
        for c in pool:
            kc, dc = orb.detectAndCompute(_gray_for_matching(c["frame"]), None)
            if dc is None or len(kc) < 12:
                continue
            matches = sorted(matcher.match(dc, ref_desc), key=lambda m: m.distance)[:300]
            if len(matches) < 12:
                continue
            src = np.float32([kc[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([ref_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            score = int(inl.sum()) if (H is not None and inl is not None) else 0
            if score > best_score or (score == best_score and best is not None and c["sharpness"] > best["sharpness"]):
                best, best_score = c, score
        if best is not None and best_score >= MIN_MATCH_INLIERS:
            return dict(best, match_inliers=best_score)
    except cv2.error:
        pass
    return fallback


def load_image_array(path: str) -> Optional[np.ndarray]:
    """Load an image OR a representative frame from a video as RGB uint8.

    Images honour EXIF orientation. Raises ``ValueError`` for an unreadable
    video; lets image-decode errors propagate so the caller can map them to a
    400 response.
    """
    if _is_video_path(path):
        frame = extract_representative_frame(path)
        if frame is None:
            raise ValueError(f"Could not read any frame from video: {path}")
        return frame
    return _load_still(path)


class _Prepared(NamedTuple):
    """Everything derived from one alignment + SSIM pass."""
    golden: np.ndarray            # RGB, capped to MAX_SIDE
    aligned: np.ndarray           # returned image warped onto golden
    ssim_map: np.ndarray          # 1.0 wherever `valid` is False
    valid: np.ndarray             # bool mask: pixels where both images have real content
    alignment: Dict[str, Any]     # {"applied": bool, "inliers": int}


# -----------------------------------------------------------------------------
# Main comparator class
# -----------------------------------------------------------------------------

class CarImageComparator:
    """AI image comparison engine using ResNet50 deep features + pixel metrics."""

    # Standard ImageNet normalization (matches the weights the model was trained with)
    _NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    _NORMALIZE_STD = [0.229, 0.224, 0.225]

    # Longest image side used for pixel metrics / heatmaps (keeps SSIM fast on 12MP photos)
    MAX_SIDE = 1024

    # Aspect ratios closer than this are treated as "the same" and simply resized.
    ASPECT_TOLERANCE = 0.03

    def __init__(self, heatmap_dir: str = "heatmaps"):
        # Pick the best available device: CUDA (Linux/Windows NVIDIA) > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(getattr(torch, "backends", None), "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.heatmap_dir = heatmap_dir
        os.makedirs(self.heatmap_dir, exist_ok=True)

        logger.info("Initializing CarImageComparator on device: %s", self.device)
        self.model, self.model_label, self.weight_source = _load_resnet50_feature_extractor(self.device)
        # An untrained network returns near-identical embeddings for ANY two
        # images (cosine ~ 1.0). Letting that into the score silently inflates
        # every trust score, so it is excluded unless real weights are loaded.
        self.semantic_reliable = self.weight_source != "untrained fallback"

        # Trust-score weights (env-tunable); normalised so they always sum to 1.
        raw = [
            self._env_float("AUDIT_W_SSIM", 0.4),
            self._env_float("AUDIT_W_SEMANTIC", 0.4),
            self._env_float("AUDIT_W_PSNR", 0.2),
        ]
        total = sum(raw)
        if total <= 0:
            raw, total = [0.4, 0.4, 0.2], 1.0
        self.w_ssim, self.w_sem, self.w_psnr = (w / total for w in raw)

        # Localised damage penalty: when `damage_full_pct` percent of the image
        # (or more) has changed, the similarity score is reduced by `max_penalty`.
        self.damage_full_pct = max(0.1, self._env_float("AUDIT_DAMAGE_FULL_PCT", 5.0))
        self.max_penalty = min(0.95, self._env_float("AUDIT_DAMAGE_MAX_PENALTY", 0.75))

        # --- Detection thresholds (env-tunable so auditors can tune sensitivity) ---
        # A pixel counts as "changed" only when its local structure differs
        # (SSIM map < ssim_change_threshold) AND its colour differs noticeably
        # (colour_diff > colour_change_threshold). Both are env-tunable so an
        # operator can make the tool stricter or looser without editing code.
        self.ssim_change_threshold = self._env_float("AUDIT_SSIM_CHANGE_THRESHOLD", 0.5)
        self.colour_change_threshold = self._env_float("AUDIT_COLOUR_CHANGE_THRESHOLD", 30.0)
        # Smallest connected-component area to count as a real region (fraction of
        # the image area; floored at 12 px so noise never survives).
        self.min_region_frac = self._env_float("AUDIT_MIN_REGION_FRAC", 0.00005)
        # PSNR is normalised by this value before being weighted into the score.
        self.psnr_full_scale = max(1.0, self._env_float("AUDIT_PSNR_FULL_SCALE", 50.0))
        # Heatmap colour scale: a (blurred, brightness-compensated) grey-level
        # difference of this size is drawn fully red. A FIXED scale (instead of
        # min-max stretching) keeps a near-identical pair looking calm.
        self.heatmap_full_scale = max(1.0, self._env_float("AUDIT_HEATMAP_FULL_SCALE", 64.0))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._NORMALIZE_MEAN, std=self._NORMALIZE_STD),
        ])

    @property
    def is_offline_ready(self) -> bool:
        """True when a cached ResNet50 weight file is available (no download needed)."""
        return _find_cached_weight_file() is not None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, default))
        except (TypeError, ValueError):
            logger.warning("Invalid value for %s; using %s", name, default)
            return default
        return value if value >= 0 else default

    # --------------------------------------------------------------------------
    # Input loading + fitting
    # --------------------------------------------------------------------------

    @staticmethod
    def _describe(kind: str, cand: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {"type": kind}
        if cand is not None:
            info.update({"frame_index": cand["index"], "time_sec": cand["time_sec"],
                         "match_inliers": cand.get("match_inliers")})
        return info

    def _load_inputs(self, golden_path: str, returned_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Decode both inputs ONCE (images or videos) as full-size RGB arrays.

        When one input is a video its frame is chosen to *match the other
        input's viewpoint* (see ``pick_best_matching_frame``) instead of being
        picked in isolation. If both are videos, the golden clip's sharpest
        frame is the reference. Returns (golden_rgb, returned_rgb, input_info).
        """
        g_vid, r_vid = _is_video_path(golden_path), _is_video_path(returned_path)

        def candidates(path: str, n: int) -> List[Dict[str, Any]]:
            cands = extract_candidate_frames(path, n)
            if not cands:
                raise ValueError(f"Could not read any frame from video: {path}")
            return cands

        if not g_vid and not r_vid:
            g, r = _load_still(golden_path), _load_still(returned_path)
            return g, r, {"golden": self._describe("image"), "returned": self._describe("image")}
        if g_vid and r_vid:
            g_best = dict(_sharpest(candidates(golden_path, 5)), match_inliers=None)
            r_best = pick_best_matching_frame(candidates(returned_path, 7), g_best["frame"])
            return g_best["frame"], r_best["frame"], {
                "golden": self._describe("video", g_best), "returned": self._describe("video", r_best)}
        if r_vid:
            g = _load_still(golden_path)
            r_best = pick_best_matching_frame(candidates(returned_path, 7), g)
            return g, r_best["frame"], {"golden": self._describe("image"), "returned": self._describe("video", r_best)}
        r = _load_still(returned_path)
        g_best = pick_best_matching_frame(candidates(golden_path, 7), r)
        return g_best["frame"], r, {"golden": self._describe("video", g_best), "returned": self._describe("image")}

    @classmethod
    def _fit_to_golden(cls, golden_rgb: np.ndarray, returned_rgb: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Bring both inputs onto one canvas (golden's aspect, longest side <= MAX_SIDE).

        Returns (golden, returned, valid_mask_uint8, warnings).

        If the aspect ratios match (within ``ASPECT_TOLERANCE``) the returned
        image is simply resized. Otherwise it is scaled UNIFORMLY to fit inside
        the canvas and centred; the uncovered border is marked invalid and
        excluded from every metric. (Stretching it - the old behaviour -
        distorted the car and was reported as widespread damage, which is the
        normal situation when a 4:3 photo is compared with a 16:9 video frame.)
        """
        g_img = Image.fromarray(golden_rgb)
        scale = min(1.0, cls.MAX_SIDE / max(g_img.size))
        gw, gh = max(1, int(g_img.width * scale)), max(1, int(g_img.height * scale))
        g = np.array(g_img.resize((gw, gh), Image.LANCZOS))

        rh, rw = returned_rgb.shape[:2]
        warnings: List[str] = []
        if abs((rw / rh) / (gw / gh) - 1.0) <= cls.ASPECT_TOLERANCE:
            r = np.array(Image.fromarray(returned_rgb).resize((gw, gh), Image.LANCZOS))
            return g, r, np.full((gh, gw), 255, np.uint8), warnings

        s = min(gw / rw, gh / rh)
        nw, nh = max(1, int(round(rw * s))), max(1, int(round(rh * s)))
        r_small = np.array(Image.fromarray(returned_rgb).resize((nw, nh), Image.LANCZOS))
        canvas = np.zeros((gh, gw, 3), np.uint8)
        valid = np.zeros((gh, gw), np.uint8)
        x0, y0 = (gw - nw) // 2, (gh - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = r_small
        valid[y0:y0 + nh, x0:x0 + nw] = 255
        warnings.append(
            f"Aspect ratio differs (golden {gw / gh:.2f}:1 vs returned {rw / rh:.2f}:1). The returned input was "
            "fitted without stretching, so only the overlapping area is compared."
        )
        return g, canvas, valid, warnings

    @classmethod
    def _resize_to_common(cls, golden_rgb: np.ndarray, returned_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backwards-compatible wrapper around ``_fit_to_golden`` (arrays only)."""
        g, r, _valid, _w = cls._fit_to_golden(golden_rgb, returned_rgb)
        return g, r

    @classmethod
    def _load_aligned_pair(cls, golden_path: str, returned_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load both inputs as RGB uint8 arrays with identical dimensions."""
        comp = cls.__new__(cls)  # _load_inputs needs no model state
        g, r, _info = comp._load_inputs(golden_path, returned_path)
        return cls._resize_to_common(g, r)

    # --------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------

    def extract_features_from_array(self, rgb: np.ndarray) -> np.ndarray:
        """2048-d ResNet50 embedding for an already-decoded RGB array."""
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        # ResNet50 with fc stripped -> shape (1, 2048, 1, 1)
        return features.cpu().numpy().flatten()

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract a 2048-d ResNet50 feature embedding for an image (or video frame)."""
        return self.extract_features_from_array(load_image_array(image_path))

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        n1 = float(np.linalg.norm(feat1))
        n2 = float(np.linalg.norm(feat2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(feat1, feat2) / (n1 * n2))

    # --------------------------------------------------------------------------
    # Alignment
    # --------------------------------------------------------------------------

    @staticmethod
    def _align_returned(golden: np.ndarray, returned: np.ndarray, valid: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Warp `returned` onto `golden` (ORB features + RANSAC homography) to
        compensate for camera shifts. Falls back to the unmodified image when
        there are too few reliable matches (e.g. already aligned shots).

        Returns (aligned_rgb, valid_mask_uint8, info). ``valid`` marks the
        pixels of `returned` that hold real content (it is warped along with
        the image), so areas that the warp had to invent - the replicated
        border - are never mistaken for damage.
        """
        h, w = golden.shape[:2]
        if valid is None:
            valid = np.full((h, w), 255, np.uint8)
        info: Dict[str, Any] = {"applied": False, "inliers": 0}
        try:
            g = cv2.cvtColor(golden, cv2.COLOR_RGB2GRAY)
            r = cv2.cvtColor(returned, cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(2000)
            kg, dg = orb.detectAndCompute(g, None)
            kr, dr = orb.detectAndCompute(r, None)
            if dg is None or dr is None or len(kg) < 30 or len(kr) < 30:
                return returned, valid, info
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(dr, dg), key=lambda m: m.distance)[:300]
            if len(matches) < 30:
                return returned, valid, info
            src = np.float32([kr[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([kg[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            if H is None or inliers is None:
                return returned, valid, info
            n_in = int(inliers.sum())
            if n_in < 25 or n_in < 0.25 * len(matches):
                return returned, valid, info
            # Reject extreme warps (bad matches): a plausible camera move has a
            # moderate scale change and almost no perspective component.
            if not (0.5 < np.linalg.det(H[:2, :2]) < 2.0):
                return returned, valid, info
            if abs(H[2, 0]) > 1e-3 or abs(H[2, 1]) > 1e-3:
                return returned, valid, info
            aligned = cv2.warpPerspective(returned, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cover = cv2.warpPerspective(valid, H, (w, h), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return aligned, cover, {"applied": True, "inliers": n_in}
        except cv2.error:
            return returned, valid, info

    # --------------------------------------------------------------------------
    # Single-pass preparation (shared by metrics, change detection, heatmap)
    # --------------------------------------------------------------------------

    def _prepare_pair(self, golden: np.ndarray, returned: np.ndarray,
                      valid: Optional[np.ndarray] = None) -> _Prepared:
        """
        Align the returned image onto the golden image and compute the SSIM map
        once. Everything else (metrics, change mask, heatmap) is derived from
        this result, so a single ``compare_images()`` call only runs ORB once
        and SSIM once.
        """
        if rgb2gray is None:  # pragma: no cover - skimage is required
            raise RuntimeError("scikit-image is required for comparison.")
        aligned, cover, info = self._align_returned(golden, returned, valid)
        gray_g = rgb2gray(golden)
        gray_r = rgb2gray(aligned)
        _, ssim_map = ssim(gray_g, gray_r, full=True, data_range=1.0)
        # Shrink the valid area a little: SSIM's 7x7 window and the warp's
        # interpolation make the pixels right at an edge unreliable.
        core = cv2.erode(cover, np.ones((9, 9), np.uint8)) > 0
        if not core.any():  # nothing overlaps; degrade gracefully instead of dividing by zero
            core = np.ones(core.shape, bool)
        ssim_map = np.where(core, ssim_map, 1.0)
        return _Prepared(golden, aligned, ssim_map, core, info)

    # --------------------------------------------------------------------------
    # Pixel-level metrics
    # --------------------------------------------------------------------------

    def compute_pixel_metrics(self, img1_path: str, img2_path: str) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """Compute SSIM, MSE, PSNR and MAD plus an SSIM difference map.

        Backwards-compatible public entry point: loads + aligns from paths and
        returns (metrics, ssim_map).
        """
        if ssim is None:
            raise RuntimeError("scikit-image is required for pixel metrics.")
        comp_g, comp_r, _info = self._load_inputs(img1_path, img2_path)
        g, r, valid, _w = self._fit_to_golden(comp_g, comp_r)
        prep = self._prepare_pair(g, r, valid)
        return self._compute_pixel_metrics(prep.golden, prep.aligned, prep.ssim_map, prep.valid), prep.ssim_map

    def _compute_pixel_metrics(
        self, golden_rgb: np.ndarray, aligned_returned_rgb: np.ndarray, ssim_map: np.ndarray,
        valid: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute metrics (over the valid pixels) from aligned arrays and a ready SSIM map."""
        if ssim is None:
            raise RuntimeError("scikit-image is required for pixel metrics.")
        if valid is None or not valid.any():
            valid = np.ones(ssim_map.shape, bool)
        ssim_score = float(ssim_map[valid].mean())
        diff = golden_rgb.astype(np.float64) - aligned_returned_rgb.astype(np.float64)
        diff = diff[valid]
        mse = float(np.mean(diff ** 2))
        # Identical images give infinite PSNR, which is not valid JSON; cap it.
        psnr = 100.0 if mse == 0 else min(100.0, float(20 * np.log10(255.0 / np.sqrt(mse))))
        mad = float(np.mean(np.abs(diff)))
        return {
            "ssim": round(ssim_score, 4),
            "mse": round(mse, 2),
            "psnr": round(psnr, 2),
            "mad": round(mad, 2),
        }

    # --------------------------------------------------------------------------
    # Localised change detection
    # --------------------------------------------------------------------------

    def detect_changes(self, golden_rgb: np.ndarray, returned_rgb: np.ndarray) -> Dict:
        """
        Find regions that really changed between two images (aligns the second
        onto the first first). Backwards-compatible public entry point;
        ``compare_images`` calls ``_detect_changes`` with data it already has.

        A pixel counts as changed only when BOTH the local structure differs
        (SSIM map) AND the colour differs noticeably, so uniform brightness
        changes (different light) are mostly ignored.
        """
        prep = self._prepare_pair(golden_rgb, returned_rgb)
        return self._detect_changes(prep.golden, prep.aligned, prep.ssim_map, prep.valid)

    def _detect_changes(
        self, golden_rgb: np.ndarray, aligned_returned_rgb: np.ndarray, ssim_map: np.ndarray,
        valid: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run change detection on aligned arrays + ready SSIM map (valid pixels only)."""
        h, w = golden_rgb.shape[:2]
        if valid is None or not valid.any():
            valid = np.ones((h, w), bool)
        valid_px = int(valid.sum())

        # Colour difference after removing the global brightness offset
        gf = golden_rgb.astype(np.float32)
        rf = aligned_returned_rgb.astype(np.float32)
        rf = rf - (rf[valid].mean(axis=0) - gf[valid].mean(axis=0))
        cdiff = np.abs(gf - rf).max(axis=2)
        cdiff = cv2.GaussianBlur(cdiff, (5, 5), 0)

        mask = (
            (ssim_map < self.ssim_change_threshold) & (cdiff > self.colour_change_threshold) & valid
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask[~valid] = 0

        n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = max(12, int(self.min_region_frac * valid_px))
        regions: List[Dict[str, float]] = []
        kept = np.zeros_like(mask)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < min_area:
                continue
            kept[_labels == i] = 255
            regions.append({
                "x": round(x / w, 4), "y": round(y / h, 4),
                "w": round(bw / w, 4), "h": round(bh / h, 4),
                "area_pct": round(100.0 * area / valid_px, 3),
            })
        regions.sort(key=lambda r: r["area_pct"], reverse=True)
        changed_pct = round(100.0 * float((kept > 0).sum()) / valid_px, 3)
        return {
            "changed_area_pct": changed_pct,
            "region_count": len(regions),
            "largest_region_pct": regions[0]["area_pct"] if regions else 0.0,
            "regions": regions[:25],
            "mask": kept,
            "aligned_returned": aligned_returned_rgb,
            "valid_area_pct": round(100.0 * valid_px / (h * w), 2),
        }

    # --------------------------------------------------------------------------
    # Heatmap generation
    # --------------------------------------------------------------------------

    def generate_heatmap(self, golden_path: str, returned_path: str, output_path: str,
                         regions: Optional[list] = None) -> Dict[str, str]:
        """
        Build a damage-localization overlay and a side-by-side composite.

        Backwards-compatible public entry point. Loads + aligns both inputs and
        draws the heatmap on the ALIGNED returned image, so the overlay always
        matches the yellow region boxes. Returns a dict with three paths:
          - overlay: heatmap blended onto the aligned returned image
          - difference: pure difference map (grayscale)
          - composite: side-by-side [golden | aligned-returned | overlay]
        """
        raw_g, raw_r, _info = self._load_inputs(golden_path, returned_path)
        g, r, valid, _w = self._fit_to_golden(raw_g, raw_r)
        prep = self._prepare_pair(g, r, valid)
        return self._generate_heatmap(prep.golden, prep.aligned, output_path, regions=regions, valid=prep.valid)

    def _generate_heatmap(
        self,
        golden_rgb: np.ndarray,
        aligned_returned_rgb: np.ndarray,
        output_path: str,
        regions: Optional[list] = None,
        valid: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """Draw the heatmap on already-aligned RGB arrays (no re-decode, no re-align)."""
        golden = cv2.cvtColor(golden_rgb, cv2.COLOR_RGB2BGR)
        returned = cv2.cvtColor(aligned_returned_rgb, cv2.COLOR_RGB2BGR)
        if valid is None or not valid.any():
            valid = np.ones(golden.shape[:2], bool)

        golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY).astype(np.float32)
        returned_gray = cv2.cvtColor(returned, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Ignore a uniform brightness change (different light / exposure), the
        # same way change detection does, so it doesn't paint the whole car.
        returned_gray = returned_gray - (returned_gray[valid].mean() - golden_gray[valid].mean())
        diff = np.abs(golden_gray - returned_gray)
        diff[~valid] = 0
        diff_blurred = cv2.GaussianBlur(diff, (21, 21), 0)
        diff_norm = np.clip(diff_blurred * (255.0 / self.heatmap_full_scale), 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(returned, 0.6, heatmap, 0.4, 0)
        overlay[~valid] = returned[~valid]  # no colour where there is nothing to compare
        # Outline every detected change region so the auditor knows where to look
        H_, W_ = overlay.shape[:2]
        for r in (regions or []):
            x0, y0 = int(r["x"] * W_), int(r["y"] * H_)
            x1, y1 = int((r["x"] + r["w"]) * W_), int((r["y"] + r["h"]) * H_)
            cv2.rectangle(overlay, (max(x0 - 4, 0), max(y0 - 4, 0)), (min(x1 + 4, W_ - 1), min(y1 + 4, H_ - 1)),
                          (0, 255, 255), 2)

        base, ext = os.path.splitext(output_path)
        ext = ext or ".png"
        diff_path = f"{base}_diff{ext}"
        composite_path = f"{base}_composite{ext}"

        cv2.imwrite(output_path, overlay)
        cv2.imwrite(diff_path, diff_norm)

        # Side-by-side composite with labels
        h, w = returned.shape[:2]
        label_h = 32
        labeled = np.full((h + label_h, w * 3, 3), 255, np.uint8)
        labeled[label_h:, :w] = golden
        labeled[label_h:, w:2 * w] = returned
        labeled[label_h:, 2 * w:3 * w] = overlay
        # Black labels on white strip
        cv2.putText(labeled, "Golden", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(labeled, "Returned (aligned)", (w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(labeled, "Heatmap Overlay", (2 * w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(composite_path, labeled)

        return {
            "overlay": output_path,
            "difference": diff_path,
            "composite": composite_path,
        }

    # --------------------------------------------------------------------------
    # Top-level comparison
    # --------------------------------------------------------------------------

    def compare_images(self, golden_path: str, returned_path: str, audit_id: Optional[str] = None) -> Dict:
        """Run a full comparison and return a structured result dict.

        Decodes each input exactly once, runs ORB/RANSAC alignment exactly once
        and computes the SSIM map exactly once; deep features, pixel metrics,
        change detection and the heatmap all reuse those arrays. Accepts images
        or videos (for a video, the frame that best matches the other input's
        viewpoint is used).
        """
        start = time.time()
        try:
            logger.info("[%s] Loading inputs...", audit_id)
            golden_raw, returned_raw, input_info = self._load_inputs(golden_path, returned_path)

            warnings: List[str] = []
            semantic_similarity: Optional[float] = None
            feature_len = 0
            if self.semantic_reliable:
                logger.info("[%s] Extracting deep features...", audit_id)
                golden_features = self.extract_features_from_array(golden_raw)
                returned_features = self.extract_features_from_array(returned_raw)
                semantic_similarity = self.cosine_similarity(golden_features, returned_features)
                feature_len = int(len(golden_features))
            else:
                warnings.append(
                    "Pretrained ResNet50 weights are not installed, so the semantic (deep-feature) similarity was "
                    "left out of the trust score. The score uses SSIM + PSNR + change analysis only. "
                    "Run `python scripts/download_model.py` once while online."
                )

            logger.info("[%s] Aligning + computing pixel metrics...", audit_id)
            golden_rgb, returned_rgb, fit_valid, fit_warnings = self._fit_to_golden(golden_raw, returned_raw)
            warnings.extend(fit_warnings)
            prep = self._prepare_pair(golden_rgb, returned_rgb, fit_valid)
            pixel_metrics = self._compute_pixel_metrics(prep.golden, prep.aligned, prep.ssim_map, prep.valid)

            logger.info("[%s] Detecting localised changes...", audit_id)
            changes = self._detect_changes(prep.golden, prep.aligned, prep.ssim_map, prep.valid)

            logger.info("[%s] Generating heatmap...", audit_id)
            heatmap_name = f"{audit_id or int(time.time())}_heatmap.png"
            heatmap_full = os.path.join(self.heatmap_dir, heatmap_name)
            heat_paths = self._generate_heatmap(prep.golden, prep.aligned, heatmap_full,
                                                regions=changes["regions"], valid=prep.valid)
            # Web-relative paths ("heatmaps/<file>") that are independent of the
            # server's working directory; served by the /heatmaps/ route.
            heat_rel = {k: f"heatmaps/{os.path.basename(v)}" for k, v in heat_paths.items()}

            # Weighted trust score (weights are env-tunable, normalised to sum to 1).
            # Without a trustworthy semantic term the remaining weights are renormalised.
            psnr_term = min(pixel_metrics["psnr"] / self.psnr_full_scale, 1.0)
            if semantic_similarity is not None:
                similarity = (self.w_ssim * pixel_metrics["ssim"] + self.w_sem * semantic_similarity
                              + self.w_psnr * psnr_term)
            else:
                denom = self.w_ssim + self.w_psnr
                if denom <= 0:
                    w_s, w_p = 0.5, 0.5
                else:
                    w_s, w_p = self.w_ssim / denom, self.w_psnr / denom
                similarity = w_s * pixel_metrics["ssim"] + w_p * psnr_term
            # Global averages dilute local damage, so scale the score down by how
            # much of the image actually changed.
            damage_index = min(changes["changed_area_pct"] / self.damage_full_pct, 1.0)
            trust_score = similarity * (1.0 - self.max_penalty * damage_index)
            trust_score = round(min(max(trust_score, 0.0), 1.0), 4)

            if not prep.alignment["applied"] and any(
                    i["type"] == "video" for i in input_info.values()):
                warnings.append("Frames could not be aligned automatically (too few matching features); "
                                "camera position differences may show up as changes.")

            condition, severity, recommendation = self._classify(trust_score)
            if changes["region_count"] and condition in ("EXCELLENT", "GOOD"):
                recommendation += (
                    f" {changes['region_count']} small changed region(s) outlined on the heatmap - "
                    "quick visual check advised."
                )
            elif changes["region_count"]:
                recommendation += f" {changes['region_count']} changed region(s) outlined on the heatmap."

            elapsed = round(time.time() - start, 3)
            return {
                "audit_id": audit_id,
                "trust_score": trust_score,
                "condition_assessment": condition,
                "severity_level": severity,
                "recommendation": recommendation,
                "warnings": warnings,
                "inputs": input_info,
                "semantic_analysis": {
                    "cosine_similarity": None if semantic_similarity is None else round(float(semantic_similarity), 4),
                    "feature_vector_length": feature_len,
                    "used_in_score": semantic_similarity is not None,
                },
                "pixel_level_metrics": pixel_metrics,
                "change_analysis": {
                    "changed_area_pct": changes["changed_area_pct"],
                    "region_count": changes["region_count"],
                    "largest_region_pct": changes["largest_region_pct"],
                    "regions": changes["regions"],
                    "similarity_before_penalty": round(float(similarity), 4),
                    "damage_index": round(float(damage_index), 4),
                    "compared_area_pct": changes["valid_area_pct"],
                },
                "alignment": {
                    "applied": bool(prep.alignment["applied"]),
                    "inliers": int(prep.alignment["inliers"]),
                },
                "heatmap_files": heat_rel,
                "metadata": {
                    "model_used": self.model_label,
                    "weight_source": self.weight_source,
                    "device": str(self.device),
                    "comparison_method": "Hybrid (Deep Features + Pixel Metrics + Heatmap)"
                                         if semantic_similarity is not None
                                         else "Pixel Metrics + Change Analysis + Heatmap (no pretrained weights)",
                    "processing_time_sec": elapsed,
                },
            }
        except Exception:  # noqa: BLE001
            logger.exception("Comparison failed for audit %s", audit_id)
            return {
                "error": "Could not process these images. Make sure both are valid, readable photos or videos.",
                "audit_id": audit_id,
                "trust_score": 0.0,
                "condition_assessment": "ERROR",
                "severity_level": 0,
                "recommendation": "Processing failed - check server logs",
            }

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _classify(self, trust_score: float) -> Tuple[str, int, str]:
        """Map a trust score to (condition, severity, recommendation).

        Thresholds are env-tunable so an operator can recalibrate for their
        own photo set without editing code.
        """
        t_excellent = self._env_float("AUDIT_THRESHOLD_EXCELLENT", 0.95)
        t_good = self._env_float("AUDIT_THRESHOLD_GOOD", 0.85)
        t_fair = self._env_float("AUDIT_THRESHOLD_FAIR", 0.70)
        t_poor = self._env_float("AUDIT_THRESHOLD_POOR", 0.50)
        # Keep ordering sane even if the operator sets odd values.
        t_excellent = min(1.0, max(t_good, t_excellent))
        t_good = min(t_excellent, max(t_fair, t_good))
        t_fair = min(t_good, max(t_poor, t_fair))
        t_poor = min(t_fair, t_poor)

        if trust_score >= t_excellent:
            return "EXCELLENT", 1, "No visible damage detected."
        if trust_score >= t_good:
            return "GOOD", 2, "Minor wear - acceptable condition."
        if trust_score >= t_fair:
            return "FAIR", 3, "Moderate differences - manual review recommended."
        if trust_score >= t_poor:
            return "POOR", 4, "Significant damage detected - inspection required."
        return "CRITICAL", 5, "Severe damage detected - immediate attention needed."
