# Car Audit Tool v2.4 - Change log

Created using Anthropic Claude. Internal draft: keep on internal versions until a human has
reviewed and verified it, and share only with people authorised to see the source material.

## Correctness
- Aspect-ratio mismatch no longer stretches the returned image (false CRITICAL on 4:3 vs 16:9);
  it is fitted, the uncovered area is excluded, and a warning is returned.
- Alignment border (replicated edge) is no longer flagged as damage: a validity mask is warped
  with the image and excluded from SSIM/PSNR/MSE/MAD, change detection and the heatmap.
- Untrained-weights fallback: deep-feature similarity (cosine ~1.0 for any pair) is excluded from
  the trust score, weights renormalised, warning returned.
- EXIF orientation is applied to photos.
- Deep features reuse the decoded arrays (each input decoded once; video decode no longer doubled).
- Heatmap uses a fixed colour scale (`AUDIT_HEATMAP_FULL_SCALE`) and ignores global brightness shift.
- Video: viewpoint-matched frame selection, frame info reported, WebM with unknown frame count handled.

## Frontend / API
- CSP: `media-src 'self' blob:` added; inline `style=""` removed (stats bars and video preview were broken).
- JSON 413 for oversized uploads; UI tolerates non-JSON error bodies; warnings and used frames shown.
- Passwords limited to 72 bytes (was HTTP 500); failed audits no longer leave uploads on disk;
  token verified once per request; PDF handles null semantic similarity.
- DB: `processed_by` migration, UTC timestamps with offset, deterministic ordering (schema v4).

## Docs corrected
v2.3 claimed a reused refresh token is rejected after rotation. It is not; tokens are revocable
via `token_version`, not single-use.

## Tests
48 tests (31 -> 48). Behavioural regressions verified to fail on v2.3.

---

## v2.3 (previous release)

This release fixes the correctness, security, performance, and
maintainability issues identified in the v2.2 review, and adds video input
support.

## Correctness

### Bug 1 - Heatmap was generated on unaligned images (highest priority)
`compare_images()` ran ORB/RANSAC alignment and computed change regions on the
aligned pair, but `generate_heatmap()` re-loaded the raw images and ran
`cv2.absdiff` on them — it never applied the homography. The yellow region
boxes came from aligned analysis while the red overlay came from the unaligned
diff, so a small camera shift produced a misleading "ghost" across the whole
car.

**Fix:** the comparator is now a single pipeline. `_prepare_pair()` aligns the
returned image and computes the SSIM map once; `_generate_heatmap()` draws the
overlay on the already-aligned returned array. A test
(`test_heatmap_uses_aligned_returned_not_raw`) verifies that a 10px camera
shift no longer flags the whole frame — only the real damage region survives.

### Bug 2 - Redundant expensive computation
Each `compare_images()` call loaded both images 3×, ran ORB alignment 2×, and
computed the full SSIM map 2× (one was discarded inside `compute_pixel_metrics`).

**Fix:** the single-pass pipeline loads each image once, aligns once, and
computes the SSIM map once. `compute_pixel_metrics`, `detect_changes`, and
`generate_heatmap` all reuse the shared aligned arrays + SSIM map via private
array-based variants (`_compute_pixel_metrics`, `_detect_changes`,
`_generate_heatmap`). The public path-based methods remain for backwards
compatibility. A test (`test_single_pass_pipeline_runs_each_step_once`)
instruments the helpers and asserts exactly one alignment pass and one SSIM
pass per audit. Per-audit latency is roughly halved (measured ~0.23 s for a
640×480 pair on CPU).

## New feature: video support

The tool previously accepted only images. It now accepts images **or** short
videos for either input:

- `models/comparator.extract_representative_frame()` samples up to 3 evenly
  spaced frames from the middle 80% of a clip and returns the sharpest one
  (highest Laplacian variance), so motion-blurred intros/outros are skipped.
- `load_image_array()` dispatches to the frame extractor for video paths and
  to Pillow for images, so the rest of the pipeline is unchanged.
- `app.py` accepts `mp4 / mov / avi / mkv / webm / m4v / wmv / 3gp / mpg / mpeg`
  in addition to the existing image types; the validator uses OpenCV's FFMPEG
  backend to confirm the file is decodable before processing.
- The frontend drop zones and batch file inputs now accept `image/*,video/*`
  and render a first-frame poster for uploaded videos.
- Tests: `test_video_upload_runs_comparison` uploads a real MP4 through the
  Flask test client and asserts a full comparison result is produced;
  `test_video_frame_extraction` exercises the extractor directly.

## Security

### Bug 4 - Refresh tokens are stateless and non-revocable
A stolen refresh token was valid for up to 7 days with no logout/revoke
endpoint, and rotation did not invalidate the old token.

**Fix:** added `token_version` (integer) to the `users` table; every JWT now
carries a `ver` claim matching it. `/api/auth/logout` (new, public-with-token)
bumps the column, which instantly invalidates the caller's outstanding access
**and** refresh tokens. Token rotation re-checks `ver` against the DB, so a
reused/old refresh token is rejected.

### Bug 5 - Password / role changes did not invalidate existing tokens
Role and password changes only took effect when the token naturally expired
(8h).

**Fix:** `update_password`, `update_user_role`, and `delete_user` all bump
`token_version`, so the change takes effect immediately. The change-password
endpoint additionally returns a fresh token pair so the caller stays signed in
on this device. A new `/api/users/<id>/role` admin endpoint exposes role
changes (previously only user creation set the role).

### Bug 6 - In-memory rate limiter
Kept as-is (documented in SECURITY.md), but the leak where one-off IP keys
accumulated forever is fixed: `_login_blocked` now drops the key once its
window is empty.

### Bug 7 - Minor hardening
- 429 responses now carry a `Retry-After` header.
- Every response gets `Content-Security-Policy`, `Strict-Transport-Security`,
  `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`.
- The JWT secret file is `chmod 0o600` on load if it already exists with looser
  perms (previously only on first creation).
- Failed logins are logged as security events (username + IP).

### Bug 12 - Double token verification
The `before_request` gate and the `login_required` decorator both verified the
token. They now share `_authenticate_request()`, which verifies signature +
expiry + type + `token_version` exactly once per request.

## Performance / scalability

### Bug 10 - Unbounded endpoints
`/api/vehicle/<vehicle_id>/audits` now honours `limit` (default 500, capped at
500) and `offset`.

### Bug 11 - Per-file size limit
`MAX_FILE_MB` (default 50) caps each uploaded file independently of the
whole-request `MAX_UPLOAD_MB`, so a single oversized video can't consume the
entire batch budget.

### Bug 9 - No production WSGI server
`waitress` is now in `requirements.txt` (cross-platform, single process keeps
the in-memory login limiter consistent). Set `WSGI=waitress` to use it, or run
`waitress-serve --listen=127.0.0.1:5000 app:app`. gunicorn with one worker is
also documented.

## Maintainability

### Bug 13 - Private function imported across modules
`/api/health` imported `_find_cached_weight_file` from `models.comparator`.
The comparator now exposes a public `is_offline_ready` property (and a public
`find_cached_weight_file` alias); the health endpoint uses the property.

### Bug 14 - Hardcoded magic numbers
The SSIM-mask threshold (`< 0.5`), colour threshold (`> 30`), minimum region
fraction, PSNR full-scale (`/50`), and the four classification thresholds are
now env-tunable: `AUDIT_SSIM_CHANGE_THRESHOLD`, `AUDIT_COLOUR_CHANGE_THRESHOLD`,
`AUDIT_MIN_REGION_FRAC`, `AUDIT_PSNR_FULL_SCALE`, `AUDIT_THRESHOLD_EXCELLENT`,
`AUDIT_THRESHOLD_GOOD`, `AUDIT_THRESHOLD_FAIR`, `AUDIT_THRESHOLD_POOR`.

### Bug 15 - Short audit_id
`uuid4().hex[:8]` gave only 32 bits. Audit IDs now use the full `uuid4().hex`
(128 bits). The DB UNIQUE constraint still catches the (now astronomically
unlikely) collision.

### Frontend
- The change-password modal said "min 6 chars" but the server requires 8;
  fixed to "min 8 chars" and the JS check now requires 8.
- The modal now saves the fresh token pair returned by the server (the old
  token is invalidated by the password change).
- The "Sign out" menu item now calls `/api/auth/logout` to revoke server-side.
- The footer version label is updated to v2.3.

## Tests

The suite grew from 14 to 31 tests. New tests cover: token revocation on
logout / password change / role change; the `Retry-After` and security headers;
per-file size limit; bounded vehicle-audits endpoint; video upload + frame
extraction; full-length audit IDs; the public `is_offline_ready` property;
env-tunable thresholds; the single-pass pipeline (alignment + SSIM each run
exactly once); and the aligned-heatmap correctness fix.
