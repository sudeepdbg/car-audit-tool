import io
import os

import numpy as np
from PIL import Image, ImageDraw

from conftest import make_png


def _damaged_png():
    img = Image.open(make_png(color=(150, 40, 40), size=(320, 200)))
    d = ImageDraw.Draw(img)
    for i in range(6):
        d.rectangle((20 + i * 45, 30, 45 + i * 45, 70), fill=(20, 20, 20))
    d.line((30, 150, 290, 120), fill=(240, 240, 240), width=5)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_local_damage_is_detected_and_lowers_score(app_module, tmp_path):
    comp = app_module.comparator
    base = tmp_path / "g.png"
    same = tmp_path / "same.png"
    dmg = tmp_path / "dmg.png"
    base.write_bytes(make_png(color=(150, 40, 40), size=(320, 200)).read())
    same.write_bytes(make_png(color=(150, 40, 40), size=(320, 200)).read())
    dmg.write_bytes(_damaged_png().read())

    ok = comp.compare_images(str(base), str(same), audit_id="t_same")
    bad = comp.compare_images(str(base), str(dmg), audit_id="t_dmg")

    assert ok["change_analysis"]["region_count"] == 0
    assert bad["change_analysis"]["changed_area_pct"] > 1.0
    assert bad["change_analysis"]["region_count"] >= 1
    assert bad["trust_score"] < ok["trust_score"] - 0.2
    assert bad["condition_assessment"] not in ("EXCELLENT", "GOOD")


def test_heatmap_uses_aligned_returned_not_raw(app_module, tmp_path):
    """Bug #1: the heatmap overlay must be drawn on the ALIGNED returned image,
    not the raw upload. With a small camera shift on a textured photo the
    aligned overlay should flag only the real damage region, not the whole
    shifted frame (which would be a misleading ghost)."""
    import cv2
    comp = app_module.comparator

    # Photo-like textured golden (lots of ORB keypoints so alignment succeeds).
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (11, 11), 0)
    cv2.rectangle(base, (60, 60), (340, 240), (220, 30, 30), 6)
    cv2.circle(base, (200, 150), 50, (30, 200, 30), -1)
    g_path = str(tmp_path / "gold.png")
    cv2.imwrite(g_path, base)

    # Returned = gold shifted 10px right + one small real damage patch.
    ret = np.zeros_like(base)
    ret[:, 10:, :] = base[:, :-10, :]
    ret[120:150, 250:290] = [255, 255, 0]
    r_path = str(tmp_path / "ret.png")
    cv2.imwrite(r_path, ret)

    out = str(tmp_path / "heat.png")
    res = comp.compare_images(g_path, r_path, audit_id="aligned_test")
    # Only the small damage patch should survive alignment + SSIM/colour gates,
    # NOT the whole frame (the bug would flag the entire shifted image).
    assert res["change_analysis"]["changed_area_pct"] < 5.0, (
        "heatmap flagged the whole frame; alignment was not applied to the overlay. "
        f"changed_area_pct={res['change_analysis']['changed_area_pct']}"
    )
    # The overlay file exists and is non-blank.
    import os
    overlay_abs = out.replace(".png", ".png")
    # The comparator writes to its configured heatmap dir, so look there.
    saved = os.path.join(comp.heatmap_dir, "aligned_test_heatmap.png")
    assert os.path.isfile(saved) and os.path.getsize(saved) > 0


def test_single_pass_pipeline_runs_each_step_once(app_module, tmp_path, monkeypatch):
    """Bug #2: a single compare_images() call runs ORB alignment and the SSIM
    map exactly once. We patch the helpers to count invocations."""
    comp = app_module.comparator
    base = tmp_path / "g.png"
    same = tmp_path / "s.png"
    base.write_bytes(make_png(color=(150, 40, 40), size=(160, 120)).read())
    same.write_bytes(make_png(color=(150, 40, 40), size=(160, 120)).read())

    align_calls = {"n": 0}
    ssim_calls = {"n": 0}
    orig_align = comp._align_returned
    orig_load = comp._load_aligned_pair

    def counting_align(g, r, *args, **kwargs):
        align_calls["n"] += 1
        return orig_align(g, r, *args, **kwargs)

    # Patch the SSIM function the module uses to count its calls.
    from models import comparator as cmp_mod
    orig_ssim = cmp_mod.ssim

    def counting_ssim(*a, **k):
        ssim_calls["n"] += 1
        return orig_ssim(*a, **k)

    monkeypatch.setattr(comp, "_align_returned", staticmethod(counting_align))
    monkeypatch.setattr(cmp_mod, "ssim", counting_ssim)
    try:
        comp.compare_images(str(base), str(same), audit_id="count_test")
    finally:
        monkeypatch.setattr(cmp_mod, "ssim", orig_ssim)

    # _align_returned may run once inside _prepare_pair (the others reuse it).
    assert align_calls["n"] == 1, f"expected 1 alignment pass, got {align_calls['n']}"
    # The SSIM map is computed exactly once (the old code computed it twice and
    # threw one away inside compute_pixel_metrics).
    assert ssim_calls["n"] == 1, f"expected 1 SSIM pass, got {ssim_calls['n']}"


def test_is_offline_ready_property(app_module):
    """Bug #13: comparator exposes a public is_offline_ready property."""
    assert hasattr(app_module.comparator, "is_offline_ready")
    assert isinstance(app_module.comparator.is_offline_ready, bool)


def test_env_tunable_thresholds(app_module, monkeypatch):
    """Bug #14: the SSIM/colour/min-area/PSNR/classification thresholds are
    env-tunable, not hard-coded magic numbers."""
    monkeypatch.setenv("AUDIT_SSIM_CHANGE_THRESHOLD", "0.3")
    monkeypatch.setenv("AUDIT_COLOUR_CHANGE_THRESHOLD", "10.0")
    monkeypatch.setenv("AUDIT_PSNR_FULL_SCALE", "40.0")
    monkeypatch.setenv("AUDIT_THRESHOLD_GOOD", "0.80")
    monkeypatch.setenv("AUDIT_THRESHOLD_FAIR", "0.60")
    from models.comparator import CarImageComparator
    c = CarImageComparator.__new__(CarImageComparator)
    # Re-run only __init__'s env reads (avoids loading the model again).
    c.w_ssim = c.w_sem = c.w_psnr = 0.0
    c.damage_full_pct = 5.0
    c.max_penalty = 0.75
    c.ssim_change_threshold = CarImageComparator._env_float("AUDIT_SSIM_CHANGE_THRESHOLD", 0.5)
    c.colour_change_threshold = CarImageComparator._env_float("AUDIT_COLOUR_CHANGE_THRESHOLD", 30.0)
    c.min_region_frac = CarImageComparator._env_float("AUDIT_MIN_REGION_FRAC", 0.00005)
    c.psnr_full_scale = CarImageComparator._env_float("AUDIT_PSNR_FULL_SCALE", 50.0)
    assert c.ssim_change_threshold == 0.3
    assert c.colour_change_threshold == 10.0
    assert c.psnr_full_scale == 40.0
    # With tuned GOOD=0.80 / FAIR=0.60 (POOR stays at default 0.50):
    # 0.85 -> GOOD, 0.70 -> FAIR, 0.55 -> POOR.
    assert c._classify(0.85)[0] == "GOOD"
    assert c._classify(0.70)[0] == "FAIR"
    assert c._classify(0.55)[0] == "POOR"


def test_video_frame_extraction(tmp_path):
    """New: extract_representative_frame returns an RGB array from a real MP4."""
    from conftest import make_mp4
    from models.comparator import extract_representative_frame
    p = make_mp4(str(tmp_path / "clip.mp4"), frames=20, size=(64, 48), box=True)
    frame = extract_representative_frame(p)
    assert frame is not None
    assert frame.shape == (48, 64, 3)
    assert frame.dtype.name == "uint8"



# ---------------------------------------------------------------------------
# v2.4 regression tests
# ---------------------------------------------------------------------------

def _scene(w=640, h=360, seed=3):
    """Textured, feature-rich 'photo' so ORB alignment has something to lock on to."""
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (9, 9), 0)
    for i in range(10):
        cv2.rectangle(img, (30 + i * 55, 60 + (i % 3) * 40), (70 + i * 55, 200), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    cv2.circle(img, (w // 2, h // 2), 60, (30, 200, 30), -1)
    return img


def test_aspect_ratio_mismatch_is_fitted_not_stretched(app_module, tmp_path):
    """A 4:3 crop of the same scene must NOT be stretched to 16:9 and reported as
    widespread damage. It is fitted, the uncovered border is excluded, and the
    result carries a warning."""
    import cv2
    comp = app_module.comparator
    gold = _scene()
    crop = gold[:, 80:560]                       # 480x360 = 4:3, same pixel scale
    cv2.imwrite(str(tmp_path / "g.png"), gold)
    cv2.imwrite(str(tmp_path / "r.png"), crop)
    res = comp.compare_images(str(tmp_path / "g.png"), str(tmp_path / "r.png"), audit_id="aspect")
    assert "error" not in res
    assert res["change_analysis"]["changed_area_pct"] < 1.0
    assert res["change_analysis"]["compared_area_pct"] < 100.0          # only the overlap is judged
    assert res["pixel_level_metrics"]["ssim"] > 0.9
    assert any("Aspect ratio" in w for w in res["warnings"])


def test_warp_border_is_not_flagged_as_damage(app_module, tmp_path):
    """Shifting the camera creates a smeared border after alignment; that border
    must not show up as changed regions on an otherwise identical scene."""
    import cv2
    comp = app_module.comparator
    gold = _scene()
    M = np.float32([[1, 0, 14], [0, 1, -9]])
    shifted = cv2.warpAffine(gold, M, (gold.shape[1], gold.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(tmp_path / "g.png"), gold)
    cv2.imwrite(str(tmp_path / "r.png"), shifted)
    res = comp.compare_images(str(tmp_path / "g.png"), str(tmp_path / "r.png"), audit_id="border")
    assert res["change_analysis"]["region_count"] == 0, res["change_analysis"]["regions"]
    assert res["change_analysis"]["changed_area_pct"] == 0.0
    assert res["alignment"]["applied"] is True


def test_exif_orientation_is_honoured(tmp_path):
    """A phone photo stored upside-down with EXIF Orientation=3 must load upright."""
    from models.comparator import load_image_array
    base = np.zeros((40, 60, 3), np.uint8)
    base[:20, :] = (255, 0, 0)                       # red top half, black bottom half
    upside_down = Image.fromarray(base).rotate(180)
    exif = Image.Exif()
    exif[0x0112] = 3
    p = tmp_path / "phone.jpg"
    upside_down.save(p, quality=100, subsampling=0, exif=exif)
    arr = load_image_array(str(p))
    assert arr[5, 30, 0] > 200 and arr[35, 30, 0] < 60   # red is on top again


def test_untrained_weights_are_excluded_from_score(app_module, tmp_path, monkeypatch):
    """An untrained network yields cosine ~1.0 for any pair; that must not be
    blended into the score. The result flags it and renormalises the weights."""
    comp = app_module.comparator
    monkeypatch.setattr(comp, "semantic_reliable", False)
    (tmp_path / "a.png").write_bytes(make_png(color=(150, 40, 40), size=(160, 120)).read())
    (tmp_path / "b.png").write_bytes(make_png(color=(150, 40, 40), size=(160, 120)).read())
    res = comp.compare_images(str(tmp_path / "a.png"), str(tmp_path / "b.png"), audit_id="nosem")
    assert res["semantic_analysis"]["cosine_similarity"] is None
    assert res["semantic_analysis"]["used_in_score"] is False
    assert any("semantic" in w.lower() for w in res["warnings"])
    assert res["trust_score"] == 1.0                      # identical pair, SSIM=1, PSNR capped


def test_reliable_semantic_term_is_used(app_module, tmp_path, monkeypatch):
    comp = app_module.comparator
    monkeypatch.setattr(comp, "semantic_reliable", True)
    (tmp_path / "a.png").write_bytes(make_png(color=(150, 40, 40), size=(160, 120)).read())
    res = comp.compare_images(str(tmp_path / "a.png"), str(tmp_path / "a.png"), audit_id="sem")
    assert res["semantic_analysis"]["used_in_score"] is True
    assert res["semantic_analysis"]["feature_vector_length"] == 2048


def test_features_do_not_redecode_inputs(app_module, tmp_path, monkeypatch):
    """Each input is decoded exactly once per audit (the docs always claimed this;
    deep features used to re-open both files, doubling video decode time)."""
    from models import comparator as cmp_mod
    comp = app_module.comparator
    monkeypatch.setattr(comp, "semantic_reliable", True)
    (tmp_path / "a.png").write_bytes(make_png(color=(150, 40, 40), size=(160, 120)).read())
    (tmp_path / "b.png").write_bytes(make_png(color=(150, 40, 40), size=(160, 120)).read())
    calls = {"n": 0}
    orig = cmp_mod._load_still

    def counting(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(cmp_mod, "_load_still", counting)
    comp.compare_images(str(tmp_path / "a.png"), str(tmp_path / "b.png"), audit_id="decode_once")
    assert calls["n"] == 2


def test_heatmap_is_calm_for_near_identical_pair(app_module, tmp_path):
    """Fixed-scale colouring: a barely visible +6 grey-level patch must stay dim.
    Min-max normalisation used to stretch whatever the largest difference was
    to full red, so a near-identical pair still showed alarming hot spots."""
    import cv2
    comp = app_module.comparator
    gold = _scene()
    ret = gold.copy()
    ret[100:140, 200:260] = np.clip(ret[100:140, 200:260].astype(np.int16) + 6, 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "g.png"), gold)
    cv2.imwrite(str(tmp_path / "r.png"), ret)
    res = comp.compare_images(str(tmp_path / "g.png"), str(tmp_path / "r.png"), audit_id="calm")
    diff_img = cv2.imread(os.path.join(comp.heatmap_dir, "calm_heatmap_diff.png"), cv2.IMREAD_GRAYSCALE)
    assert diff_img.max() < 60, f"a +6 level patch was stretched to {diff_img.max()}/255"
    assert res["change_analysis"]["region_count"] == 0


def test_video_frame_matches_other_inputs_viewpoint(tmp_path):
    """pick_best_matching_frame must choose the frame showing the SAME view as
    the reference even when a frame of a different view is sharper."""
    from models.comparator import pick_best_matching_frame
    view_a, view_b, view_c = _scene(seed=1), _scene(seed=2), _scene(seed=3)
    cands = [
        {"frame": cv2_rgb(view_a), "index": 0, "time_sec": 0.0, "sharpness": 900.0},   # sharpest, wrong view
        {"frame": cv2_rgb(view_b), "index": 10, "time_sec": 1.0, "sharpness": 500.0},  # right view
        {"frame": cv2_rgb(view_c), "index": 20, "time_sec": 2.0, "sharpness": 700.0},
    ]
    best = pick_best_matching_frame(cands, cv2_rgb(view_b))
    assert best["index"] == 10 and best["match_inliers"] >= 15


def cv2_rgb(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_video_without_frame_count_still_yields_a_frame(tmp_path, monkeypatch):
    """Browser-recorded WebM often reports frame_count 0/-1; fall back to a scan."""
    import cv2
    from conftest import make_mp4
    from models import comparator as cmp_mod
    p = make_mp4(str(tmp_path / "clip.mp4"), frames=40, size=(64, 48), box=True)

    real_cap = cv2.VideoCapture

    class NoCountCap:
        def __init__(self, path):
            self._c = real_cap(path)

        def get(self, prop):
            return -1.0 if prop == cv2.CAP_PROP_FRAME_COUNT else self._c.get(prop)

        def __getattr__(self, name):
            return getattr(self._c, name)

    monkeypatch.setattr(cmp_mod.cv2, "VideoCapture", NoCountCap)
    cands = cmp_mod.extract_candidate_frames(p, sample_count=4)
    assert 1 <= len(cands) <= 4
    assert len({c["index"] for c in cands}) == len(cands)       # not just frame 0 repeated
    assert max(c["index"] for c in cands) > 0
