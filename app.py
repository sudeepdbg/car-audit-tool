"""
Car Audit Tool - Flask web server with a JWT-protected REST API.

Compares a "golden" (before) image against a "returned" (after) image and
produces a trust score, metrics and damage-localisation heatmaps.

Security notes:
- Every /api/ route except login/refresh/health requires a JWT.
- Uploaded images and heatmaps are NOT public; they require a valid token.
- Binds to 127.0.0.1 by default. Set HOST=0.0.0.0 only behind a reverse proxy
  with TLS.
- Login attempts are rate limited per IP+username (in-memory).
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from functools import wraps
from typing import Any, Callable, Deque, Dict, Optional
from xml.sax.saxutils import escape as xml_escape

from PIL import Image, UnidentifiedImageError

from flask import (
    Flask, jsonify, render_template, request, send_from_directory, send_file,
    g,
)
from werkzeug.utils import secure_filename

from models.comparator import CarImageComparator
from models.database import AuditDatabase
from models.auth import AuthManager, MIN_PASSWORD_LENGTH, password_problem, seed_default_admin


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("car_audit_tool.app")


# -----------------------------------------------------------------------------
# App + configuration
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "heatmaps")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
# Inputs may be images OR short videos. A representative frame is extracted
# from a video automatically (see models/comparator.extract_representative_frame).
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "wmv", "3gp", "mpg", "mpeg"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
MAX_BATCH_PAIRS = int(os.environ.get("MAX_BATCH_PAIRS", "20"))
MAX_NOTES_LEN = 5000
# Guard against decompression bombs (default Pillow limit only warns).
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(60_000_000)))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Whole-request size cap (multipart body). A separate per-file cap below
# stops a single oversized video from consuming the entire budget.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024
# Per-file size cap (bytes). Defaults to 50 MB; videos may need a larger value
# than images, so set MAX_FILE_MB higher if you accept long clips. A request
# with N files is also bounded by MAX_CONTENT_LENGTH, so for a 20-pair batch
# you may need to raise MAX_UPLOAD_MB accordingly.
MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_MB", "50")) * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

comparator = CarImageComparator(heatmap_dir=HEATMAP_FOLDER)
DATA_DIR = os.environ.get("AUDIT_DATA_DIR") or os.path.join(BASE_DIR, "database")
audit_db = AuditDatabase(db_path=os.path.join(DATA_DIR, "audit_history.db"))

# Auth setup: HMAC secret persisted under database/jwt_secret.key
auth = AuthManager(
    secret_path=os.path.join(DATA_DIR, "jwt_secret.key"),
    secret_env=os.environ.get("JWT_SECRET"),
)
# First start only: create an admin. If DEFAULT_ADMIN_PASSWORD is unset, a random
# password is generated and printed once to the console.
seed_default_admin(
    audit_db,
    username=os.environ.get("DEFAULT_ADMIN_USERNAME", "admin"),
    password=os.environ.get("DEFAULT_ADMIN_PASSWORD") or None,
)

# Public endpoints that don't require a JWT.
PUBLIC_PATHS = {"/", "/login", "/api/auth/login", "/api/auth/refresh", "/api/auth/logout", "/api/health"}
PUBLIC_PREFIXES = ("/static/",)


def _has_video_support() -> bool:
    """True if OpenCV was built with a video-decode backend (FFMPEG/GSTREAMER).

    Used by the startup banner and the health endpoint so operators know whether
    video uploads will work on their platform before relying on the feature.
    """
    try:
        import cv2
        info = cv2.getBuildInformation()
        return ("FFMPEG" in info) or ("GSTREAMER" in info)
    except Exception:  # pragma: no cover - cv2 is a hard dependency
        return False


# -----------------------------------------------------------------------------
# Login rate limiting (in-memory, per process)
# -----------------------------------------------------------------------------

LOGIN_MAX_FAILURES = int(os.environ.get("LOGIN_MAX_FAILURES", "5"))
LOGIN_WINDOW_SEC = int(os.environ.get("LOGIN_WINDOW_SEC", "900"))
_login_failures: Dict[str, Deque[float]] = defaultdict(deque)
_login_lock = threading.Lock()


def _login_key(username: str) -> str:
    return f"{request.remote_addr}|{username.lower()}"


def _login_blocked(key: str) -> bool:
    now = time.time()
    with _login_lock:
        q = _login_failures[key]
        while q and now - q[0] > LOGIN_WINDOW_SEC:
            q.popleft()
        # Drop the key entirely once the window is empty so _login_failures
        # never accumulates dead entries (slow memory leak for one-off IPs).
        if not q:
            _login_failures.pop(key, None)
            return False
        return len(q) >= LOGIN_MAX_FAILURES


def _login_retry_after(key: str) -> int:
    """Seconds until the oldest failure for `key` falls out of the window.
    Returns 0 when the key isn't blocked (used for the Retry-After header)."""
    now = time.time()
    with _login_lock:
        q = _login_failures.get(key)
        if not q:
            return 0
        return max(1, int(LOGIN_WINDOW_SEC - (now - q[0]) + 1))


def _login_record_failure(key: str, username: str) -> None:
    with _login_lock:
        _login_failures[key].append(time.time())
    # Log as a security event so SIEM/audit trails can pick it up.
    logger.warning(
        "Failed login: username=%r ip=%s", username, request.remote_addr,
    )


def _login_clear(key: str) -> None:
    with _login_lock:
        _login_failures.pop(key, None)


# A strict-ish CSP: same-origin only, no inline scripts/styles (the templates
# load JS/CSS from /static), no remote frames or plugins. Images served from
# the app are blob URLs created from same-origin fetches, so img-src stays
# same-origin + the inline blob: scheme that those object URLs use. media-src
# needs blob: too, because the upload drop zone decodes a chosen video through
# a blob: URL to draw its preview poster.
_CSP_HEADER = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' blob: data:; "
    "media-src 'self' blob:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    # HSTS only makes sense behind HTTPS; harmless on localhost HTTP.
    resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    resp.headers.setdefault("Content-Security-Policy", _CSP_HEADER)
    if request.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.errorhandler(413)
def _too_large(_err):
    """Oversized upload: answer in JSON (the UI parses every response as JSON;
    Flask's default is an HTML page, which surfaced as a cryptic parse error)."""
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({
        "error": f"Upload too large: the whole request may be at most {limit_mb} MB "
                 f"(raise MAX_UPLOAD_MB / MAX_FILE_MB for longer videos).",
        "trust_score": 0.0,
    }), 413


def _int_arg(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(value, hi))


def _ext_of(filename: str) -> str:
    """Lowercased extension without the dot (empty string if none)."""
    return (os.path.splitext(filename)[1].lower().lstrip(".")) if "." in filename else ""


def _is_video_filename(filename: str) -> bool:
    return _ext_of(filename) in ALLOWED_VIDEO_EXTENSIONS


def _valid_image(path: str) -> bool:
    """Confirm the saved file really is an image Pillow can decode."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        return False


def _valid_video(path: str) -> bool:
    """Confirm the saved file is a video OpenCV can actually open + decode."""
    # Local import to keep app import cheap for tests that don't upload videos.
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    try:
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


def _valid_upload(path: str, is_video: bool) -> bool:
    return _valid_video(path) if is_video else _valid_image(path)


def _save_upload(file_storage, prefix: str, audit_id: str) -> Optional[str]:
    """Save an uploaded image or video under uploads/.

    Enforces the per-file size cap (``MAX_FILE_BYTES``) before writing, and
    validates the saved file is actually decodable (image via Pillow, video via
    OpenCV). Returns the stored filename or None if invalid/too large.
    """
    # Per-file size check on top of the whole-request MAX_CONTENT_LENGTH so a
    # single oversized video can't consume the entire batch budget.
    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size <= 0 or size > MAX_FILE_BYTES:
        logger.warning("Rejected upload %r: size=%s bytes (max %s)", file_storage.filename, size, MAX_FILE_BYTES)
        return None

    is_video = _is_video_filename(file_storage.filename or "")
    name = secure_filename(f"{prefix}_{audit_id}_{file_storage.filename}")
    path = os.path.join(app.config["UPLOAD_FOLDER"], name)
    file_storage.save(path)
    if not _valid_upload(path, is_video):
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    return name


def _remove_quietly(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except OSError:
            logger.warning("Could not remove %s", p)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------------------------------------------------------
# Auth helpers + decorator
# -----------------------------------------------------------------------------

def _extract_bearer() -> Optional[str]:
    """Pull the bearer token out of the Authorization header."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[7:].strip()


def current_user() -> Optional[Dict[str, Any]]:
    """Return the authenticated user dict from the request context, or None."""
    return getattr(g, "user", None)


def _set_g_user(user: Dict[str, Any]) -> None:
    g.user = {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "display_name": user.get("display_name"),
    }


def _authenticate_request() -> Optional[Any]:
    """Run ``_verify_request_token`` at most once per request.

    The ``before_request`` gate and the ``login_required`` decorator both call
    this; the second call reuses the first result instead of decoding the JWT
    and hitting SQLite again.
    """
    if getattr(g, "_auth_checked", False):
        return getattr(g, "_auth_error", None)
    err = _verify_request_token()
    g._auth_checked = True
    g._auth_error = err
    return err


def _verify_request_token() -> Optional[Any]:
    """Shared token verification used by both the ``before_request`` gate and
    the ``login_required`` decorator (so a token is verified exactly once per
    request, not twice).

    Verifies the JWT signature/expiry/type AND that the user's current
    ``token_version`` matches the token's ``ver`` claim - so a bumped version
    (password change, role change, logout, user deletion) immediately
    invalidates older tokens before their natural expiry.

    On success, populates ``g.user`` and returns None. On failure, returns the
    JSON 401/403 response to send. If no bearer token is present, returns None
    without touching ``g.user`` (so public endpoints and pre-auth probes work).
    """
    token = _extract_bearer()
    if not token:
        return None
    claims = auth.verify_token(token, expected_type="access")
    if not claims:
        return jsonify({"error": "Invalid or expired token", "code": "bad_token"}), 401
    try:
        uid = int(claims["sub"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid or expired token", "code": "bad_token"}), 401
    user = audit_db.get_user_by_id(uid)
    if not user:
        return jsonify({"error": "User no longer exists", "code": "no_user"}), 401
    # Token-version check: reject if the DB version was bumped after this token.
    try:
        if int(claims.get("ver", 0)) != int(user.get("token_version", 1) or 1):
            return jsonify({"error": "Token has been revoked", "code": "bad_token"}), 401
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or expired token", "code": "bad_token"}), 401
    _set_g_user(user)
    return None


def login_required(role: Optional[str] = None) -> Callable:
    """Decorator that requires a valid JWT (and optionally a specific role)."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            err = _authenticate_request()
            if err is not None:
                return err
            user = current_user()
            if role and user["role"] != role and user["role"] != "admin":
                return jsonify({"error": "Insufficient permissions", "code": "forbidden"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return deco


@app.before_request
def _auth_gate() -> Optional[Any]:
    """Block any non-public API endpoint that lacks a valid JWT."""
    path = request.path

    # Verify once (signature + token_version) and populate g.user. The result
    # is None on success (with g.user set) or a 401 response on failure.
    err = _authenticate_request()

    # Public pages, login/refresh/logout/health and static assets never need a
    # token. For those we just leave g.user as whatever the optional check set.
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return None

    # Everything else (the API, /uploads/, /heatmaps/) needs a valid user.
    if err is not None:
        return err
    if getattr(g, "user", None) is None:
        return jsonify({"error": "Authentication required", "code": "no_token"}), 401
    return None


# -----------------------------------------------------------------------------
# Routes - UI
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("index.html")


# -----------------------------------------------------------------------------
# Routes - auth
# -----------------------------------------------------------------------------

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    key = _login_key(username)
    if _login_blocked(key):
        resp = jsonify({"error": "Too many failed attempts. Try again later.", "code": "rate_limited"}), 429
        resp[0].headers["Retry-After"] = str(_login_retry_after(key))
        return resp
    user = audit_db.get_user_by_username(username)
    if not user or not AuthManager.verify_password(password, user["password_hash"]):
        _login_record_failure(key, username)
        return jsonify({"error": "Invalid username or password"}), 401
    _login_clear(key)
    access, refresh = auth.issue_tokens({
        "id": user["id"], "username": user["username"], "role": user["role"],
        "token_version": user.get("token_version", 1),
    })
    audit_db.update_last_login(user["id"])
    return jsonify({
        "access_token": access,
        "refresh_token": refresh,
        "user": {
            "id": user["id"], "username": user["username"], "role": user["role"],
            "display_name": user.get("display_name"),
        },
    }), 200


@app.route("/api/auth/refresh", methods=["POST"])
def auth_refresh():
    body = request.get_json(silent=True) or {}
    refresh_token = body.get("refresh_token")
    if not refresh_token:
        return jsonify({"error": "refresh_token required"}), 400
    uid = auth.user_id_from_refresh(refresh_token)
    user = audit_db.get_user_by_id(uid) if uid is not None else None
    pair = auth.refresh_access(refresh_token, user) if user else None
    if not pair:
        return jsonify({"error": "Invalid or expired refresh token"}), 401
    access, new_refresh = pair
    return jsonify({"access_token": access, "refresh_token": new_refresh}), 200


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """Revoke the caller's tokens by bumping their token_version.

    The endpoint accepts the access token (preferred, in the Authorization
    header) or a refresh token in the JSON body; either way the user's
    ``token_version`` is incremented, which invalidates every previously issued
    access AND refresh token for that user before their natural expiry. Returns
    200 even when no token is given so the UI can call it on every sign-out
    without caring about edge cases.
    """
    token = _extract_bearer()
    claims = None
    if token:
        claims = auth.verify_token(token, expected_type="access")
    # Fallback: a refresh token supplied in the body (used by the UI when only
    # the refresh token is still available, e.g. after the access token expired).
    if claims is None:
        body = request.get_json(silent=True) or {}
        rt = body.get("refresh_token")
        if rt:
            claims = auth.verify_token(rt, expected_type="refresh")
    if claims:
        try:
            uid = int(claims["sub"])
            audit_db.bump_token_version(uid)
        except (KeyError, ValueError, TypeError):
            pass
    return jsonify({"ok": True}), 200


@app.route("/api/auth/me", methods=["GET"])
@login_required()
def auth_me():
    return jsonify({"user": current_user()}), 200


@app.route("/api/auth/change-password", methods=["POST"])
@login_required()
def auth_change_password():
    body = request.get_json(silent=True) or {}
    current = body.get("current_password") or ""
    new = body.get("new_password") or ""
    problem = password_problem(new)
    if problem:
        return jsonify({"error": "New " + problem[0].lower() + problem[1:]}), 400
    u = current_user()
    full = audit_db.get_user_by_id(u["id"])
    if not AuthManager.verify_password(current, full["password_hash"]):
        return jsonify({"error": "Current password is incorrect"}), 401
    # update_password bumps token_version, so the caller's current token (and
    # every other token for this user) is invalidated; the UI must re-login.
    audit_db.update_password(u["id"], AuthManager.hash_password(new))
    # Issue a fresh pair so the caller stays signed in on this device.
    full = audit_db.get_user_by_id(u["id"])
    access, refresh = auth.issue_tokens({
        "id": full["id"], "username": full["username"], "role": full["role"],
        "token_version": full.get("token_version", 1),
    })
    return jsonify({"ok": True, "access_token": access, "refresh_token": refresh}), 200


@app.route("/api/users", methods=["GET", "POST"])
@login_required(role="admin")
def users_endpoint():
    if request.method == "GET":
        return jsonify({"users": audit_db.list_users()}), 200
    # POST - create new user
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    role = body.get("role", "user")
    if role not in ("admin", "user"):
        return jsonify({"error": "Role must be 'admin' or 'user'"}), 400
    if not username:
        return jsonify({"error": "Username required"}), 400
    problem = password_problem(password)
    if problem:
        return jsonify({"error": problem}), 400
    if audit_db.get_user_by_username(username):
        return jsonify({"error": "Username already taken"}), 409
    uid = audit_db.create_user(
        username=username,
        password_hash=AuthManager.hash_password(password),
        role=role,
        display_name=body.get("display_name"),
    )
    return jsonify({"ok": True, "id": uid}), 201


@app.route("/api/users/<int:user_id>", methods=["DELETE"])
@login_required(role="admin")
def users_delete(user_id: int):
    if user_id == current_user()["id"]:
        return jsonify({"error": "You cannot delete your own account"}), 400
    ok = audit_db.delete_user(user_id)
    if not ok:
        return jsonify({"error": "Cannot delete the last admin user"}), 400
    return jsonify({"ok": True}), 200


@app.route("/api/users/<int:user_id>/role", methods=["POST"])
@login_required(role="admin")
def users_change_role(user_id: int):
    """Change a user's role. Bumps token_version so the change takes effect
    immediately instead of after the 8h access-token expiry (Bug #5)."""
    body = request.get_json(silent=True) or {}
    role = body.get("role")
    if role not in ("admin", "user"):
        return jsonify({"error": "Role must be 'admin' or 'user'"}), 400
    if user_id == current_user()["id"]:
        return jsonify({"error": "Use the admin UI to change your own role carefully"}), 400
    target = audit_db.get_user_by_id(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404
    if target["role"] == "admin" and role == "user":
        # Prevent demoting the last admin to a non-admin role.
        remaining = audit_db.list_users()
        if sum(1 for u in remaining if u["role"] == "admin") <= 1:
            return jsonify({"error": "Cannot demote the last admin user"}), 400
    if not audit_db.update_user_role(user_id, role):
        return jsonify({"error": "Could not update role"}), 400
    return jsonify({"ok": True}), 200


# -----------------------------------------------------------------------------
# Routes - health & meta
# -----------------------------------------------------------------------------

@app.route("/api/health")
def health():
    """Public liveness/offline-readiness check; extra details only when signed in."""
    user = current_user()
    payload: Dict[str, Any] = {
        "status": "ok",
        "offline_ready": comparator.is_offline_ready,
        "auth_enabled": True,
        "auth_user": user.get("username") if user else None,
        "auth_role": user.get("role") if user else None,
    }
    if user:
        payload.update({
            "model": comparator.model_label,
            "weight_source": comparator.weight_source,
            "device": str(comparator.device),
            "audits_count": audit_db.get_statistics()["total_audits"],
        })
    return jsonify(payload)


# -----------------------------------------------------------------------------
# Routes - single compare
# -----------------------------------------------------------------------------

@app.route("/api/compare", methods=["POST"])
def compare_images():
    golden_file = request.files.get("golden_image")
    returned_file = request.files.get("returned_image")

    if not golden_file or not returned_file:
        return jsonify({
            "error": "Both golden_image and returned_image files are required.",
            "trust_score": 0.0,
        }), 400
    if golden_file.filename == "" or returned_file.filename == "":
        return jsonify({"error": "No selected files.", "trust_score": 0.0}), 400
    if not (allowed_file(golden_file.filename) and allowed_file(returned_file.filename)):
        return jsonify({
            "error": "Invalid file type. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS)),
            "trust_score": 0.0,
        }), 400

    audit_id = uuid.uuid4().hex
    golden_filename = _save_upload(golden_file, "golden", audit_id)
    returned_filename = _save_upload(returned_file, "returned", audit_id)
    if not golden_filename or not returned_filename:
        _remove_quietly(
            *(os.path.join(app.config["UPLOAD_FOLDER"], n) for n in (golden_filename, returned_filename) if n)
        )
        return jsonify({"error": "One of the files is not a valid image or video, or exceeds the size limit.",
                         "trust_score": 0.0}), 400
    golden_path = os.path.join(app.config["UPLOAD_FOLDER"], golden_filename)
    returned_path = os.path.join(app.config["UPLOAD_FOLDER"], returned_filename)

    result = comparator.compare_images(golden_path, returned_path, audit_id=audit_id)
    user = current_user() or {}
    result.update({
        "vehicle_id": request.form.get("vehicle_id", ""),
        "customer_name": request.form.get("customer_name", ""),
        "rental_start_date": request.form.get("rental_start_date", ""),
        "rental_end_date": request.form.get("rental_end_date", ""),
        "golden_image_path": golden_filename,
        "returned_image_path": returned_filename,
        "processed_by": user.get("username", "system"),
    })

    if "error" in result:
        # Nothing is saved for a failed audit, so don't leave the customer's
        # photos behind on disk.
        _remove_quietly(golden_path, returned_path)
        return jsonify(result), 422
    if not audit_db.save_audit(result):
        result["warning"] = "Result could not be saved to history."
    return jsonify(result), 200


# -----------------------------------------------------------------------------
# Routes - batch compare
# -----------------------------------------------------------------------------

@app.route("/api/batch-compare", methods=["POST"])
def batch_compare():
    """
    Accepts multipart/form-data with repeating fields:
        golden_image_<i>, returned_image_<i>, vehicle_id_<i>
    for i = 0, 1, 2, ...  (until no more pairs are found, max MAX_BATCH_PAIRS)
    """
    pairs = []
    user = current_user() or {}
    i = 0
    while i < MAX_BATCH_PAIRS:
        gfile = request.files.get(f"golden_image_{i}")
        rfile = request.files.get(f"returned_image_{i}")
        if gfile is None and rfile is None:
            break
        if gfile is None or rfile is None:
            pairs.append({"pair_index": i, "error": "Missing golden or returned image", "trust_score": 0.0})
            i += 1
            continue
        if not allowed_file(gfile.filename) or not allowed_file(rfile.filename):
            pairs.append({"pair_index": i, "error": "Invalid file type", "trust_score": 0.0})
            i += 1
            continue
        audit_id = uuid.uuid4().hex
        gname = _save_upload(gfile, "golden", audit_id)
        rname = _save_upload(rfile, "returned", audit_id)
        if not gname or not rname:
            _remove_quietly(*(os.path.join(app.config["UPLOAD_FOLDER"], n) for n in (gname, rname) if n))
            pairs.append({"pair_index": i, "error": "File is not a valid image/video, or exceeds the size limit",
                          "trust_score": 0.0})
            i += 1
            continue
        result = comparator.compare_images(
            os.path.join(app.config["UPLOAD_FOLDER"], gname),
            os.path.join(app.config["UPLOAD_FOLDER"], rname),
            audit_id=audit_id,
        )
        result["pair_index"] = i
        result["vehicle_id"] = request.form.get(f"vehicle_id_{i}", "")
        result["golden_image_path"] = gname
        result["returned_image_path"] = rname
        result["processed_by"] = user.get("username", "system")
        if "error" not in result:
            audit_db.save_audit(result)
        else:
            _remove_quietly(os.path.join(app.config["UPLOAD_FOLDER"], gname),
                            os.path.join(app.config["UPLOAD_FOLDER"], rname))
        pairs.append(result)
        i += 1

    successful = sum(1 for r in pairs if "error" not in r)
    failed = len(pairs) - successful
    avg = (
        sum(r.get("trust_score", 0) for r in pairs if "error" not in r) / successful
        if successful else 0.0
    )
    return jsonify({
        "total_pairs": len(pairs),
        "successful": successful,
        "failed": failed,
        "avg_trust_score": round(avg, 4),
        "results": pairs,
    }), 200


# -----------------------------------------------------------------------------
# Routes - audit history
# -----------------------------------------------------------------------------

@app.route("/api/audits", methods=["GET"])
def get_audits():
    limit = _int_arg("limit", 50, 1, 500)
    offset = _int_arg("offset", 0, 0, 10_000_000)
    audits = audit_db.get_all_audits(limit=limit, offset=offset)
    return jsonify({"audits": audits, "count": len(audits)}), 200


@app.route("/api/audit/<audit_id>", methods=["GET"])
def get_audit(audit_id: str):
    audit = audit_db.get_audit(audit_id)
    if not audit:
        return jsonify({"error": "Audit not found"}), 404
    return jsonify(audit), 200


@app.route("/api/audit/<audit_id>/notes", methods=["POST"])
def update_notes(audit_id: str):
    body = request.get_json(silent=True) or {}
    notes = body.get("notes", "")
    if not isinstance(notes, str) or len(notes) > MAX_NOTES_LEN:
        return jsonify({"error": f"Notes must be text up to {MAX_NOTES_LEN} characters"}), 400
    if not audit_db.update_notes(audit_id, notes):
        return jsonify({"error": "Audit not found"}), 404
    return jsonify({"ok": True}), 200


@app.route("/api/audit/<audit_id>", methods=["DELETE"])
@login_required(role="admin")
def delete_audit(audit_id: str):
    """Admin only. Also removes the stored images and heatmaps for the audit."""
    audit = audit_db.get_audit(audit_id)
    if not audit or not audit_db.delete_audit(audit_id):
        return jsonify({"ok": False, "error": "Audit not found"}), 404
    files = [os.path.join(UPLOAD_FOLDER, os.path.basename(audit.get(k) or ""))
             for k in ("golden_image_path", "returned_image_path")]
    heat = audit.get("heatmap_files")
    if isinstance(heat, dict):
        files += [os.path.join(HEATMAP_FOLDER, os.path.basename(v)) for v in heat.values() if v]
    _remove_quietly(*files)
    return jsonify({"ok": True}), 200


@app.route("/api/vehicle/<vehicle_id>/audits", methods=["GET"])
def get_vehicle_audits(vehicle_id: str):
    # Bound the response so a vehicle with thousands of audits doesn't return
    # an unbounded payload. limit defaults to 500 and is capped at 500.
    limit = _int_arg("limit", 500, 1, 500)
    offset = _int_arg("offset", 0, 0, 10_000_000)
    audits = audit_db.get_audits_by_vehicle(vehicle_id, limit=limit, offset=offset)
    return jsonify({"vehicle_id": vehicle_id, "audits": audits, "count": len(audits)}), 200


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    return jsonify(audit_db.get_statistics()), 200


# -----------------------------------------------------------------------------
# Routes - PDF report export
# -----------------------------------------------------------------------------

@app.route("/api/audit/<audit_id>/export/pdf", methods=["GET"])
def export_pdf(audit_id: str):
    audit = audit_db.get_audit(audit_id)
    if not audit:
        return jsonify({"error": "Audit not found"}), 404

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
        )
    except ImportError:
        return jsonify({"error": "reportlab not installed - add to requirements.txt"}), 500

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Heading1"], textColor=colors.HexColor("#4f46e5"))
    body_style = styles["BodyText"]
    mono_style = ParagraphStyle("mono", parent=styles["Code"], fontSize=9, leading=12)

    story = []
    story.append(Paragraph("Car Audit Report", title_style))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f"<b>Audit ID:</b> {xml_escape(str(audit.get('audit_id')))}", body_style))
    story.append(Paragraph(f"<b>Vehicle ID:</b> {xml_escape(str(audit.get('vehicle_id') or 'N/A'))}", body_style))
    story.append(Paragraph(f"<b>Customer:</b> {xml_escape(str(audit.get('customer_name') or 'N/A'))}", body_style))
    story.append(Paragraph(f"<b>Created:</b> {xml_escape(str(audit.get('created_at')))}", body_style))
    story.append(Spacer(1, 5 * mm))

    score = audit.get("trust_score") or 0
    score_color = (
        "#16a34a" if audit.get("condition_assessment") == "EXCELLENT"
        else "#65a30d" if audit.get("condition_assessment") == "GOOD"
        else "#ca8a04" if audit.get("condition_assessment") == "FAIR"
        else "#dc2626" if audit.get("condition_assessment") == "POOR"
        else "#991b1b"
    )
    story.append(Paragraph(
        f"<b>Trust Score:</b> <font color='{score_color}'>{score:.4f}</font> "
        f"({xml_escape(str(audit.get('condition_assessment')))})",
        body_style,
    ))
    story.append(Paragraph(f"<b>Recommendation:</b> {xml_escape(str(audit.get('recommendation')))}", body_style))
    story.append(Spacer(1, 5 * mm))

    def _cell(key: str) -> str:
        value = audit.get(key)
        return "n/a" if value is None else str(value)

    table_data = [
        ["Metric", "Value"],
        ["SSIM", _cell("ssim")],
        ["MSE", _cell("mse")],
        ["PSNR (dB)", _cell("psnr")],
        ["MAD", _cell("mad")],
        ["Changed area (%)", _cell("changed_area_pct")],
        ["Changed regions", _cell("region_count")],
        ["Semantic Similarity", "n/a (not used)" if audit.get("semantic_similarity") is None else _cell("semantic_similarity")],
        ["Processing Time (s)", _cell("processing_time_sec")],
        ["Model", str(audit.get("weight_source") or "n/a")],
    ]
    t = Table(table_data, colWidths=[55 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
    ]))
    story.append(t)
    story.append(Spacer(1, 5 * mm))

    # Embed heatmap if available
    heat = audit.get("heatmap_files") or {}
    composite_path = None
    if isinstance(heat, dict):
        rel = heat.get("composite") or heat.get("overlay")
        if rel:
            composite_path = os.path.join(HEATMAP_FOLDER, os.path.basename(rel))
    if composite_path and os.path.isfile(composite_path):
        try:
            img = RLImage(composite_path, width=160 * mm, height=60 * mm, kind="proportional")
            story.append(Paragraph("<b>Heatmap comparison:</b>", body_style))
            story.append(img)
        except Exception:
            pass

    if audit.get("notes"):
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("<b>Auditor notes:</b>", body_style))
        story.append(Paragraph(xml_escape(str(audit.get("notes"))), body_style))

    doc.build(story)
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"audit_{audit_id}.pdf",
        mimetype="application/pdf",
    )


# -----------------------------------------------------------------------------
# Routes - static asset serving (uploads, heatmaps)
# -----------------------------------------------------------------------------

@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/heatmaps/<path:filename>")
def serve_heatmap(filename: str):
    return send_from_directory(HEATMAP_FOLDER, filename)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("DEBUG", "0") == "1"
    if debug and host not in ("127.0.0.1", "localhost"):
        logger.warning("DEBUG=1 with a non-local HOST exposes the Werkzeug debugger; forcing debug off.")
        debug = False
    print("=" * 60)
    print(" Car Audit Tool v2.4")
    print("=" * 60)
    print(f" Model:   {comparator.model_label}")
    print(f" Weights: {comparator.weight_source}")
    if not comparator.semantic_reliable:
        print(" WARNING: no pretrained weights - deep-feature similarity is EXCLUDED from scores.")
        print("          Run `python scripts/download_model.py` once while online.")
    print(f" Device:  {comparator.device}")
    print(f" Video:   {'supported' if _has_video_support() else 'opencv-video backend unavailable'}")
    print("=" * 60)
    print(f" Listening on http://{host}:{port}")
    print("=" * 60)
    # In production prefer waitress (cross-platform, single process so the
    # in-memory rate limiter still works): `waitress-serve --listen=127.0.0.1:5000 app:app`
    # or gunicorn (Linux/macOS): `gunicorn -w 1 -b 127.0.0.1:5000 app:app`.
    if not debug and os.environ.get("WSGI", "0") == "waitress":
        try:
            from waitress import serve as waitress_serve
            waitress_serve(app, host=host, port=port)
        except ImportError:
            logger.info("WSGI=waitress set but waitress not installed; falling back to app.run().")
            app.run(debug=debug, host=host, port=port)
    else:
        app.run(debug=debug, host=host, port=port)
