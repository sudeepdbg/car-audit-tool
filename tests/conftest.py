import io
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_png(color=(120, 120, 120), size=(96, 64), box=None) -> io.BytesIO:
    img = Image.new("RGB", size, color)
    if box:
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def make_mp4(path, frames=30, size=(96, 64), box=None, fps=10) -> str:
    """Write a tiny real MP4 video to ``path`` and return the path.

    Uses OpenCV's FFMPEG backend. Each frame is a flat colour with an optional
    moving red box, so the sharpest-frame picker in the comparator has
    something real to look at. Falls back to writing raw frames into a Motion
    JPEG AVI if mp4v is unavailable on the test platform.
    """
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        # Fallback: MJPEG in an .avi container (always available).
        path = path[:-4] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, size)
    try:
        for i in range(frames):
            frame = np.full((size[1], size[0], 3), 120, dtype=np.uint8)
            if box:
                x = (i * 2) % (size[0] - 20)
                cv2.rectangle(frame, (x, 10), (x + 15, 25), (0, 0, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    """Import app.py once, pointed at a throwaway data dir (never the real database/)."""
    tmp = tmp_path_factory.mktemp("data")
    os.environ["AUDIT_DATA_DIR"] = str(tmp)
    os.environ["DEFAULT_ADMIN_PASSWORD"] = "test-password-1"
    os.environ["JWT_SECRET"] = "x" * 48
    import app as app_mod

    for attr, name in (("UPLOAD_FOLDER", "uploads"), ("HEATMAP_FOLDER", "heatmaps")):
        d = tmp / name
        d.mkdir()
        setattr(app_mod, attr, str(d))
    app_mod.app.config["UPLOAD_FOLDER"] = str(tmp / "uploads")
    app_mod.comparator.heatmap_dir = str(tmp / "heatmaps")
    from models.database import AuditDatabase
    app_mod.audit_db = AuditDatabase(db_path=str(tmp / "test.db"))
    from models.auth import seed_default_admin
    seed_default_admin(app_mod.audit_db, "admin", "test-password-1")
    app_mod.app.config["TESTING"] = True
    return app_mod


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture()
def admin_headers(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "test-password-1"})
    assert r.status_code == 200, r.get_json()
    return {"Authorization": "Bearer " + r.get_json()["access_token"]}


@pytest.fixture()
def admin_refresh_token(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "test-password-1"})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["refresh_token"]
