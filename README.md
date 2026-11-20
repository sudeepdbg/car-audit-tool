# Car Audit Tool

[![CI](https://github.com/YOUR_GITHUB_USERNAME/car-audit-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/car-audit-tool/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A self-hosted web app that compares a **"golden"** (before-rental) photo with a
**"returned"** (after-rental) photo and produces a similarity **trust score**,
image metrics, and a **difference heatmap** to help a human reviewer spot
changes quickly. Includes batch mode, audit history, statistics, PDF reports,
and JWT-based multi-user login. After a one-time model download it runs fully offline.

Both inputs may be **images** (PNG / JPG / WEBP / BMP / GIF / TIFF) or **short
videos** (MP4 / MOV / AVI / MKV / WEBM / ...). A representative (sharpest) frame
is extracted from a video automatically and run through the same pipeline, so
auditors can upload a clip instead of a still when that is what they have.
When one input is a video, the frame that best matches the *other* input's
viewpoint (ORB feature matches) is used, and the result reports which frame.
Inputs with different aspect ratios (e.g. a 4:3 photo vs a 16:9 video frame)
are fitted without stretching and only the overlapping area is compared.

> **Important - what this is and isn't.** The trust score is a *similarity
> heuristic* (SSIM + ResNet50 embedding similarity + PSNR). It is **not** a
> trained damage detector and cannot tell a scratch from a shadow, a different
> angle, or different lighting. Use it to **triage** image pairs for human
> review, never as the sole basis for charging a customer. Photos should be
> taken from the same angle and in similar light for meaningful results. Thin
> single scratches may only produce a small flagged region and a GOOD score -
> always look at the outlined regions.

## Features

- Hybrid comparison: SSIM, MSE, PSNR, MAD + ResNet50 cosine similarity
- Difference heatmap: overlay, difference map, and side-by-side composite, all
  drawn on the **aligned** returned image so the red overlay always lines up
  with the yellow region boxes
- Single-pass pipeline: each image is loaded once, ORB/RANSAC alignment runs
  once, and the SSIM map is computed once per audit (~halves per-audit latency)
- Image OR video input (a representative frame is extracted from a video)
- Batch comparison (up to 20 pairs per request by default)
- Audit history with search, auditor notes, and one-page PDF export
- Statistics dashboard (trend, condition distribution, lowest-scoring vehicles)
- Login with bcrypt password hashing, JWT access/refresh tokens, admin/user roles
- Revocable tokens (not single-use: a refresh token stays valid until it expires or `token_version` is bumped): a `token_version` column is stamped into every JWT, so
  password changes, role changes, deletes, and logout all invalidate the
  user's outstanding tokens immediately (no 8h/7d grace window)
- Per-file + whole-request upload size limits
- Light/dark UI, drag-and-drop uploads
- Runs offline once weights are cached

## Quick start

Requires Python 3.10 - 3.12.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/car-audit-tool.git
cd car-audit-tool

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One-time, needs internet: cache the ResNet50 weights (~100 MB)
python scripts/download_model.py

python app.py
```

Open <http://127.0.0.1:5000>. On the **first start** an admin user is created
and a **random password is printed once in the console** - copy it, sign in as
`admin`, then change it (user menu -> Change password). Passwords must be 8-72 bytes (a bcrypt limit). To choose your own, set
`DEFAULT_ADMIN_PASSWORD` (8+ chars) before the first start.

Smaller CPU-only PyTorch install:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

Without cached weights the app falls back to an **untrained** network. Its deep-feature
similarity is meaningless (cosine ~1.0 for any pair), so it is **excluded from the trust
score** (remaining weights renormalised) and the result carries a warning. Run
`scripts/download_model.py` once to enable it.

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` only behind a TLS reverse proxy |
| `PORT` | `5000` | HTTP port |
| `DEBUG` | `0` | `1` enables Flask debug (ignored unless HOST is local) |
| `WSGI` | `0` | `waitress` to serve via waitress (single process, keeps the in-memory login limiter consistent); otherwise the dev server |
| `JWT_SECRET` | auto-generated | 32+ random chars; otherwise stored in `database/jwt_secret.key` (chmod 0o600) |
| `DEFAULT_ADMIN_USERNAME` | `admin` | First-run admin username |
| `DEFAULT_ADMIN_PASSWORD` | random, printed once | First-run admin password (8+ chars) |
| `LOGIN_MAX_FAILURES` / `LOGIN_WINDOW_SEC` | `5` / `900` | Login rate limit per IP+username |
| `MAX_UPLOAD_MB` | `50` | Max request size (multipart body) |
| `MAX_FILE_MB` | `50` | Max size per uploaded file (separate from the whole-request cap) |
| `MAX_BATCH_PAIRS` | `20` | Max pairs per batch request |
| `MAX_IMAGE_PIXELS` | `60000000` | Reject images larger than this (decompression-bomb guard) |
| `AUDIT_W_SSIM` / `AUDIT_W_SEMANTIC` / `AUDIT_W_PSNR` | `0.4` / `0.4` / `0.2` | Trust-score weights (normalised to sum to 1) |
| `AUDIT_DAMAGE_FULL_PCT` | `5` | % of image changed at which the full damage penalty applies (lower = stricter) |
| `AUDIT_DAMAGE_MAX_PENALTY` | `0.75` | Maximum fraction of the score removed by the damage penalty |
| `AUDIT_SSIM_CHANGE_THRESHOLD` | `0.5` | SSIM-map value below which a pixel's structure is "changed" |
| `AUDIT_COLOUR_CHANGE_THRESHOLD` | `30` | Per-channel colour difference above which a pixel's colour is "changed" |
| `AUDIT_MIN_REGION_FRAC` | `0.00005` | Smallest change region kept, as a fraction of image area |
| `AUDIT_HEATMAP_FULL_SCALE` | `64` | Grey-level difference drawn fully red on the heatmap (fixed scale, not min-max) |
| `AUDIT_PSNR_FULL_SCALE` | `50` | PSNR (dB) normalised to this value before being weighted into the score |
| `AUDIT_THRESHOLD_EXCELLENT` / `_GOOD` / `_FAIR` / `_POOR` | `0.95` / `0.85` / `0.70` / `0.50` | Trust-score condition boundaries |
| `TORCH_HOME` | `~/.cache/torch` | Model weight cache location |
| `AUDIT_DATA_DIR` | `./database` | Where the SQLite DB and JWT secret file are stored |
| `LOG_LEVEL` | `INFO` | Logging level |

## How the score works

```
similarity  = w1 * SSIM + w2 * cosine_similarity + w3 * min(PSNR / 50, 1)     (default 0.4 / 0.4 / 0.2)
damage_idx  = min(changed_area_% / AUDIT_DAMAGE_FULL_PCT, 1)                  (default full penalty at 5 % changed)
trust_score = similarity * (1 - AUDIT_DAMAGE_MAX_PENALTY * damage_idx)        (default max penalty 0.75)
```

Global metrics (SSIM, embedding similarity) are averages over the whole photo,
so localised damage on an otherwise identical photo barely moves them. The
**change analysis** step therefore also finds the regions that really changed:
the returned image is aligned to the golden one (ORB + homography, to absorb
small camera shifts), then a pixel counts as "changed" only when both its local
structure (SSIM map) and its colour differ noticeably (global brightness offset
removed). Changed regions are outlined on the heatmap and drive the penalty.

Both images are resized to a common size (longest side capped at 1024 px). Areas the alignment
warp had to invent (borders) are excluded from all metrics. The semantic term only counts when
pretrained weights are loaded.

| Score | Condition | Severity |
| --- | --- | --- |
| 0.95 - 1.00 | EXCELLENT | 1 |
| 0.85 - 0.95 | GOOD | 2 |
| 0.70 - 0.85 | FAIR | 3 |
| 0.50 - 0.70 | POOR | 4 |
| 0.00 - 0.50 | CRITICAL | 5 |

These thresholds are heuristics and have not been calibrated on a labelled
damage dataset. Tune them for your own photos.

## REST API

All routes except those marked public require `Authorization: Bearer <access_token>`.

| Method | Path | Access | Description |
| --- | --- | --- | --- |
| GET | `/api/health` | public | Status + offline readiness (more detail when signed in) |
| POST | `/api/auth/login` | public | Get access (8h) + refresh (7d) tokens |
| POST | `/api/auth/refresh` | public | Exchange a refresh token |
| POST | `/api/auth/logout` | public (token optional) | Bump `token_version` so the caller's access + refresh tokens are revoked immediately |
| GET | `/api/auth/me` | user | Current user |
| POST | `/api/auth/change-password` | user | Change own password; bumps `token_version` and returns a fresh token pair so the caller stays signed in |
| GET/POST | `/api/users` | admin | List / create users |
| DELETE | `/api/users/<id>` | admin | Delete a user (not yourself, not the last admin) |
| POST | `/api/users/<id>/role` | admin | Change a user's role; bumps `token_version` so the change takes effect immediately |
| POST | `/api/compare` | user | Single comparison (multipart: `golden_image`, `returned_image`, optional `vehicle_id`, `customer_name`, `rental_start_date`, `rental_end_date`). Images OR short videos are accepted |
| POST | `/api/batch-compare` | user | Fields `golden_image_<i>`, `returned_image_<i>`, `vehicle_id_<i>` |
| GET | `/api/audits?limit=&offset=` | user | History |
| GET | `/api/audit/<id>` | user | One audit |
| POST | `/api/audit/<id>/notes` | user | Save notes (max 5000 chars) |
| DELETE | `/api/audit/<id>` | admin | Delete audit and its stored images |
| GET | `/api/audit/<id>/export/pdf` | user | PDF report |
| GET | `/api/vehicle/<vehicle_id>/audits?limit=&offset=` | user | Audits for a vehicle (limit capped at 500) |
| GET | `/api/statistics` | user | Dashboard data |
| GET | `/uploads/<file>`, `/heatmaps/<file>` | user | Stored images (token required) |

## Security and privacy

- Uploaded photos and heatmaps require authentication.
- Passwords are bcrypt-hashed; there is **no hard-coded default password**.
- Login is rate limited (in-memory, per process); failed attempts are logged as
  security events. A 429 response carries a `Retry-After` header.
- Uploads are validated as real images / videos and size-limited per file and
  per request.
- JWTs are stateless **but revocable**: every token carries a `ver` claim that
  matches the user's `token_version` column. Password changes, role changes,
  deletes, and `/api/auth/logout` all bump that column, which invalidates the
  user's outstanding access + refresh tokens immediately (no 8h / 7d grace).
  Refresh tokens are **not** single-use: rotating one does not invalidate the old
  one before it expires; use logout / password change to revoke.
  The web UI keeps tokens in browser storage; always serve over HTTPS if
  exposed beyond localhost.
- Responses carry `Content-Security-Policy`, `X-Frame-Options: DENY`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and HSTS.
- Rental photos can contain personal data (faces, plates). You are responsible
  for retention and consent under the laws that apply to you.
- For production use a real WSGI server. The recommended single-process option
  (keeps the in-memory login limiter consistent) is waitress, which works on
  Windows too:
  `WSGI=waitress python app.py` or `waitress-serve --listen=127.0.0.1:5000 app:app`.
  On Linux/macOS you can also use gunicorn with one worker:
  `gunicorn -w 1 -b 127.0.0.1:5000 app:app`. For more throughput with multiple
  workers, move the login limiter to Redis or the proxy layer.

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Project layout

```
app.py                  Flask server + REST API
models/comparator.py    Feature extraction, metrics, heatmaps
models/database.py      SQLite audit history + users
models/auth.py          JWT, bcrypt, first-run admin seeding
templates/ static/      Web UI
scripts/download_model.py   One-time weight caching
tests/                  pytest suite
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```

Contributions are welcome via pull request. Please add tests for new behaviour.

## License and third-party notes

Released under the [MIT License](LICENSE).

This project depends on third-party packages under their own licenses
(Flask, PyTorch/torchvision, OpenCV, scikit-image, ReportLab, PyJWT, bcrypt, Pillow).
The ResNet50 weights are downloaded from torchvision at setup time and are
not included in this repository; they were trained on ImageNet, so review the
ImageNet terms of access if you use this commercially.
