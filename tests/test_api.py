from conftest import make_png


def test_api_requires_auth(client):
    assert client.get("/api/audits").status_code == 401
    assert client.get("/api/statistics").status_code == 401


def test_uploads_and_heatmaps_are_not_public(client):
    assert client.get("/uploads/anything.png").status_code == 401
    assert client.get("/heatmaps/anything.png").status_code == 401


def test_health_is_minimal_when_anonymous(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert "device" not in data and "model" not in data


def test_login_rejects_bad_password(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "nope"})
    assert r.status_code == 401


def test_login_rate_limit(client, app_module):
    app_module._login_failures.clear()
    for _ in range(app_module.LOGIN_MAX_FAILURES):
        client.post("/api/auth/login", json={"username": "ghost", "password": "bad"})
    r = client.post("/api/auth/login", json={"username": "ghost", "password": "bad"})
    assert r.status_code == 429
    app_module._login_failures.clear()


def test_refresh_token_cannot_be_used_as_access(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "test-password-1"})
    refresh = r.get_json()["refresh_token"]
    assert client.get("/api/audits", headers={"Authorization": "Bearer " + refresh}).status_code == 401


def test_rejects_non_image_upload(client, admin_headers):
    import io
    r = client.post(
        "/api/compare",
        data={
            "golden_image": (io.BytesIO(b"not an image"), "a.png"),
            "returned_image": (io.BytesIO(b"nope"), "b.png"),
        },
        headers=admin_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_compare_save_notes_pdf_delete(client, admin_headers):
    r = client.post(
        "/api/compare",
        data={
            "golden_image": (make_png(), "golden.png"),
            "returned_image": (make_png(box=(10, 10, 40, 40)), "returned.png"),
            "vehicle_id": "<b>CAR-1</b>",
            "customer_name": "Test & Co",
        },
        headers=admin_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert 0.0 <= body["trust_score"] <= 1.0
    assert body["heatmap_files"]["overlay"].startswith("heatmaps/")
    audit_id = body["audit_id"]

    # protected image is retrievable with a token
    img = client.get("/" + body["heatmap_files"]["overlay"], headers=admin_headers)
    assert img.status_code == 200

    # notes
    assert client.post(f"/api/audit/{audit_id}/notes", json={"notes": "scratch <i>left</i>"},
                       headers=admin_headers).status_code == 200
    assert client.post(f"/api/audit/{audit_id}/notes", json={"notes": "x" * 6000},
                       headers=admin_headers).status_code == 400

    # PDF handles markup-like user text without crashing
    pdf = client.get(f"/api/audit/{audit_id}/export/pdf", headers=admin_headers)
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"

    # delete (admin)
    assert client.delete(f"/api/audit/{audit_id}", headers=admin_headers).status_code == 200
    assert client.get(f"/api/audit/{audit_id}", headers=admin_headers).status_code == 404


def test_identical_images_give_valid_json(client, admin_headers):
    r = client.post(
        "/api/compare",
        data={"golden_image": (make_png(), "g.png"), "returned_image": (make_png(), "r.png")},
        headers=admin_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert r.get_json()["pixel_level_metrics"]["psnr"] <= 100.0


def test_different_sized_images(client, admin_headers):
    r = client.post(
        "/api/compare",
        data={"golden_image": (make_png(size=(120, 80)), "g.png"),
              "returned_image": (make_png(size=(60, 40)), "r.png")},
        headers=admin_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 200


def test_pagination_bad_input_does_not_500(client, admin_headers):
    assert client.get("/api/audits?limit=abc&offset=-5", headers=admin_headers).status_code == 200


def test_cannot_delete_self(client, admin_headers):
    me = client.get("/api/auth/me", headers=admin_headers).get_json()["user"]
    assert client.delete(f"/api/users/{me['id']}", headers=admin_headers).status_code == 400


def test_short_password_rejected(client, admin_headers):
    r = client.post("/api/users", json={"username": "bob", "password": "short"}, headers=admin_headers)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# New tests for the v2.3 fixes
# ---------------------------------------------------------------------------

def test_logout_invalidates_access_token(client, admin_headers):
    """Bug #4: /api/auth/logout bumps token_version so the old access token
    is immediately rejected (not just after the 8h expiry)."""
    r = client.post("/api/auth/logout", headers=admin_headers)
    assert r.status_code == 200
    # The token we just used to log out is now invalid.
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 401


def test_logout_invalidates_refresh_token(client, admin_refresh_token):
    """Bug #4: logout also kills the refresh token, not just the access one."""
    r = client.post("/api/auth/logout", json={"refresh_token": admin_refresh_token})
    assert r.status_code == 200
    # Refreshing with the old refresh token must fail (ver claim is stale).
    assert client.post("/api/auth/refresh",
                       json={"refresh_token": admin_refresh_token}).status_code == 401


def test_change_password_invalidates_old_tokens(client, admin_headers):
    """Bug #5: changing the password bumps token_version, so the token used
    for the change is rejected afterwards. The endpoint returns a fresh pair.
    Restores the original password at the end so session-scoped fixtures keep
    working for the rest of the suite."""
    r = client.post("/api/auth/change-password",
                    json={"current_password": "test-password-1", "new_password": "new-pass-123"},
                    headers=admin_headers)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert "access_token" in body and "refresh_token" in body
    # The old token (used for the change) is now invalid.
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 401
    # The new pair works, and we can use it for a real request.
    new_headers = {"Authorization": "Bearer " + body["access_token"]}
    assert client.get("/api/auth/me", headers=new_headers).status_code == 200
    # The new refresh token also works.
    assert client.post("/api/auth/refresh",
                       json={"refresh_token": body["refresh_token"]}).status_code == 200
    # Restore the original password for the rest of the session.
    restore = client.post("/api/auth/change-password",
                          json={"current_password": "new-pass-123", "new_password": "test-password-1"},
                          headers=new_headers)
    assert restore.status_code == 200


def test_role_change_invalidates_tokens(client, admin_headers):
    """Bug #5: changing a user's role bumps their token_version, so any token
    they hold becomes invalid immediately (no 8h grace window)."""
    # Create a second user, then promote them - their (nonexistent) tokens
    # would be invalidated. We check the endpoint succeeds and returns 200.
    create = client.post("/api/users",
                         json={"username": "tester", "password": "tester-pass-1", "role": "user"},
                         headers=admin_headers)
    assert create.status_code == 201
    uid = create.get_json()["id"]
    # Log in as the tester so they hold a real token.
    login = client.post("/api/auth/login", json={"username": "tester", "password": "tester-pass-1"})
    assert login.status_code == 200
    tester_headers = {"Authorization": "Bearer " + login.get_json()["access_token"]}
    # Tester can hit a protected endpoint.
    assert client.get("/api/audits", headers=tester_headers).status_code == 200
    # Admin promotes tester; their token is now stale.
    assert client.post(f"/api/users/{uid}/role", json={"role": "admin"},
                       headers=admin_headers).status_code == 200
    # The tester's old token is rejected (ver mismatch).
    assert client.get("/api/audits", headers=tester_headers).status_code == 401


def test_rate_limit_returns_retry_after(client, app_module):
    """Bug #7: a 429 response carries a Retry-After header."""
    app_module._login_failures.clear()
    for _ in range(app_module.LOGIN_MAX_FAILURES):
        client.post("/api/auth/login", json={"username": "ghost2", "password": "bad"})
    r = client.post("/api/auth/login", json={"username": "ghost2", "password": "bad"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert int(r.headers["Retry-After"]) >= 1
    app_module._login_failures.clear()


def test_security_headers_present(client):
    """Bug #7: CSP and HSTS headers are applied to every response."""
    r = client.get("/api/health")
    assert "Content-Security-Policy" in r.headers
    assert "X-Frame-Options" in r.headers
    assert "Strict-Transport-Security" in r.headers
    assert "nosniff" in r.headers["X-Content-Type-Options"]


def test_per_file_size_limit_rejects_oversized(client, admin_headers, app_module, tmp_path, monkeypatch):
    """Bug #11: a single file larger than MAX_FILE_BYTES is rejected even when
    the whole request is under MAX_CONTENT_LENGTH. We lower the per-file cap
    below the size of any real test image so even a small PNG is rejected."""
    monkeypatch.setattr(app_module, "MAX_FILE_BYTES", 50)
    from conftest import make_png
    r = client.post(
        "/api/compare",
        data={
            "golden_image": (make_png(), "golden.png"),
            "returned_image": (make_png(), "returned.png"),
        },
        headers=admin_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "valid image" in r.get_json()["error"].lower()


def test_vehicle_audits_endpoint_is_bounded(client, admin_headers, app_module):
    """Bug #10: /api/vehicle/<id>/audits honours limit= and caps it at 500."""
    r = client.get("/api/vehicle/CAR-X/audits?limit=10000", headers=admin_headers)
    assert r.status_code == 200
    # The endpoint never returns more than 500 rows even when asked for 10000.
    assert r.get_json()["count"] <= 500


def test_video_upload_runs_comparison(client, admin_headers, tmp_path):
    """New video support: a short MP4 is accepted, decoded, and compared."""
    from conftest import make_mp4
    g_vid = make_mp4(str(tmp_path / "g.mp4"), frames=24, box=False)
    r_vid = make_mp4(str(tmp_path / "r.mp4"), frames=24, box=True)
    with open(g_vid, "rb") as fg, open(r_vid, "rb") as fr:
        r = client.post(
            "/api/compare",
            data={
                "golden_image": (fg, "golden.mp4"),
                "returned_image": (fr, "returned.mp4"),
                "vehicle_id": "VID-1",
            },
            headers=admin_headers,
            content_type="multipart/form-data",
        )
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert 0.0 <= body["trust_score"] <= 1.0
    assert body["heatmap_files"]["overlay"].startswith("heatmaps/")
    assert body["vehicle_id"] == "VID-1"


def test_rejects_non_video_non_image_upload(client, admin_headers):
    """A file with a disallowed extension is rejected with 400."""
    import io
    r = client.post(
        "/api/compare",
        data={
            "golden_image": (io.BytesIO(b"not anything useful"), "golden.exe"),
            "returned_image": (io.BytesIO(b"nope"), "returned.txt"),
        },
        headers=admin_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_audit_id_is_full_length(client, admin_headers):
    """Bug #15: audit_id uses the full uuid4().hex (32 chars), not 8."""
    r = client.post(
        "/api/compare",
        data={"golden_image": (make_png(), "g.png"), "returned_image": (make_png(), "r.png")},
        headers=admin_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    audit_id = r.get_json()["audit_id"]
    assert len(audit_id) == 32
    assert all(c in "0123456789abcdef" for c in audit_id)


def test_health_uses_public_offline_ready(client):
    """Bug #13: /api/health no longer imports a private comparator function."""
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "offline_ready" in r.get_json()



# ---------------------------------------------------------------------------
# v2.4 regression tests
# ---------------------------------------------------------------------------

def test_oversized_upload_returns_json_413(client, app_module, admin_headers, monkeypatch):
    """Flask's default 413 is an HTML page; the UI parses every response as JSON."""
    import io
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 2048)
    r = client.post(
        "/api/compare",
        data={"golden_image": (io.BytesIO(b"x" * 20000), "g.png"),
              "returned_image": (io.BytesIO(b"x" * 20000), "r.png")},
        headers=admin_headers, content_type="multipart/form-data",
    )
    assert r.status_code == 413
    assert r.is_json and "too large" in r.get_json()["error"].lower()


def test_overlong_password_is_rejected_not_500(client, admin_headers):
    """bcrypt only takes 72 bytes (bcrypt>=5 raises); this used to be an HTTP 500."""
    r = client.post("/api/users", json={"username": "longpw", "password": "x" * 100}, headers=admin_headers)
    assert r.status_code == 400
    assert "72" in r.get_json()["error"]
    r = client.post("/api/auth/change-password",
                    json={"current_password": "test-password-1", "new_password": "y" * 100},
                    headers=admin_headers)
    assert r.status_code == 400


def test_csp_allows_blob_media_but_not_inline_styles(client):
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "media-src 'self' blob:" in csp            # client-side video poster preview
    assert "'unsafe-inline'" not in csp               # ...and the strict policy is kept


def test_frontend_has_no_csp_blocked_inline_styles():
    """style="..." attributes are dropped by `style-src 'self'`, which silently
    broke the statistics bars. Keep them out of the template and the JS."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, "templates", "index.html"), encoding="utf-8").read()
    assert 'style="' not in html
    js_lines = open(os.path.join(root, "static", "js", "main.js"), encoding="utf-8").read().splitlines()
    offenders = [l for l in js_lines if 'style="' in l and not l.lstrip().startswith(("//", "*"))]
    assert not offenders, offenders


def test_failed_comparison_does_not_leave_uploads_behind(client, app_module, admin_headers, monkeypatch):
    import os
    folder = app_module.app.config["UPLOAD_FOLDER"]
    before = set(os.listdir(folder))
    monkeypatch.setattr(app_module.comparator, "compare_images",
                        lambda *a, **k: {"error": "boom", "trust_score": 0.0, "audit_id": k.get("audit_id")})
    r = client.post(
        "/api/compare",
        data={"golden_image": (make_png(), "g.png"), "returned_image": (make_png(), "r.png")},
        headers=admin_headers, content_type="multipart/form-data",
    )
    assert r.status_code == 422
    assert set(os.listdir(folder)) == before


def test_token_is_verified_once_per_request(client, app_module, admin_headers, monkeypatch):
    calls = {"n": 0}
    orig = app_module.auth.verify_token

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(app_module.auth, "verify_token", counting)
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 200   # gate + decorator
    assert calls["n"] == 1


def test_database_upgrades_old_schema_and_marks_timestamps_utc(tmp_path):
    """A pre-v2 audits table (no processed_by etc.) must be migrated so saving works,
    and created_at must carry a UTC offset so browsers don't read it as local time."""
    import sqlite3
    from models.database import AuditDatabase
    path = str(tmp_path / "old.db")
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE audits (id INTEGER PRIMARY KEY AUTOINCREMENT, audit_id TEXT UNIQUE NOT NULL,
        vehicle_id TEXT, customer_name TEXT, rental_start_date TEXT, rental_end_date TEXT, trust_score REAL,
        condition_assessment TEXT, severity_level INTEGER, recommendation TEXT, semantic_similarity REAL,
        ssim REAL, mse REAL, psnr REAL, mad REAL, heatmap_files TEXT, golden_image_path TEXT,
        returned_image_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    con.commit()
    con.close()
    db = AuditDatabase(path)
    assert db.save_audit({"audit_id": "a1", "trust_score": 0.9, "condition_assessment": "GOOD"}) is True
    assert db.save_audit({"audit_id": "a2", "trust_score": 0.8, "condition_assessment": "FAIR"}) is True
    row = db.get_audit("a1")
    assert row["created_at"].endswith("+00:00")
    # same-second inserts come back newest-first, deterministically
    assert [a["audit_id"] for a in db.get_all_audits()] == ["a2", "a1"]


def test_compare_response_reports_inputs_and_warnings(client, admin_headers):
    r = client.post(
        "/api/compare",
        data={"golden_image": (make_png(size=(160, 90)), "g.png"),
              "returned_image": (make_png(size=(120, 90)), "r.png")},       # 16:9 vs 4:3
        headers=admin_headers, content_type="multipart/form-data",
    )
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["inputs"]["golden"]["type"] == "image"
    assert any("Aspect ratio" in w for w in body["warnings"])
    # PDF export must cope with a null semantic similarity
    pdf = client.get(f"/api/audit/{body['audit_id']}/export/pdf", headers=admin_headers)
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")
