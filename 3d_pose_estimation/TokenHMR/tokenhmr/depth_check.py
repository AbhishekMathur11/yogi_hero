"""Compare a TokenHMR mesh to a metric depth map.

HMR camera depth is often biased, so predicted Z is shifted to match the
median RealSense depth, then we report median absolute error in metres.
"""

import numpy as np


def check_pose_depth(
    pred_smpl_params=None,
    pred_cam=None,
    depth=None,
    person_mask=None,
    vertices=None,
    cam_t=None,
    focal_length=None,
    img_size=None,
    subsample=24,
    mae_ok=0.20,
):
    """Return a dict, or None if depth or the mesh is missing.

    depth: (H, W) metres, aligned to the RGB frame.
    vertices: (V, 3) SMPL verts in HMR camera space (no translation).
    cam_t: (3,) full-image camera translation from cam_crop_to_full.
    focal_length: pixels, same scale as the overlay.
    img_size: (width, height) of the RGB frame.
    """
    del pred_smpl_params, pred_cam  # kept so old callers still match
    if depth is None or vertices is None or cam_t is None or focal_length is None:
        return None

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim == 3:
        verts = verts[0]
    cam_t = np.asarray(cam_t, dtype=np.float32).reshape(3)

    pts = verts[:: max(1, int(subsample))] + cam_t
    z = pts[:, 2]
    h, w = depth.shape
    if img_size is None:
        img_w, img_h = float(w), float(h)
    else:
        img_w, img_h = float(img_size[0]), float(img_size[1])
    fx = fy = float(focal_length)
    u = fx * pts[:, 0] / np.clip(z, 1e-6, None) + img_w / 2.0
    v = fy * pts[:, 1] / np.clip(z, 1e-6, None) + img_h / 2.0
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 0.3) & (z < 8.0)
    if person_mask is not None:
        mask = np.asarray(person_mask).astype(bool)
        if mask.shape[:2] != depth.shape:
            raise ValueError("person_mask must match depth shape")
        inside = inside & mask[vi.clip(0, h - 1), ui.clip(0, w - 1)]
    if int(inside.sum()) < 25:
        return {"ok": False, "score": 0.0, "mae_m": None, "n": int(inside.sum()), "reason": "too_few_points"}

    meas = depth[vi[inside], ui[inside]]
    valid = np.isfinite(meas) & (meas > 0.3) & (meas < 8.0)
    n = int(valid.sum())
    if n < 25:
        return {"ok": False, "score": 0.0, "mae_m": None, "n": n, "reason": "too_few_depth"}

    pred_z = z[inside][valid]
    meas_z = meas[valid]
    bias = float(np.median(pred_z) - np.median(meas_z))
    pred_z = pred_z - np.median(pred_z) + np.median(meas_z)
    mae = float(np.median(np.abs(pred_z - meas_z)))
    score = float(np.clip(1.0 - mae / 0.25, 0.0, 1.0))
    return {
        "ok": mae <= mae_ok,
        "score": round(score, 3),
        "mae_m": round(mae, 3),
        "bias_m": round(bias, 3),
        "n": n,
        "reason": None,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    verts = rng.normal(0, 0.3, size=(200, 3)).astype(np.float32)
    verts[:, 2] += 2.0
    cam_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    h, w, f = 120, 160, 200.0
    depth = np.zeros((h, w), dtype=np.float32)
    pts = verts + cam_t
    u = np.rint(f * pts[:, 0] / pts[:, 2] + w / 2).astype(int)
    v = np.rint(f * pts[:, 1] / pts[:, 2] + h / 2).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth[v[ok], u[ok]] = pts[ok, 2]
    good = check_pose_depth(depth=depth, vertices=verts, cam_t=cam_t, focal_length=f, img_size=(w, h), subsample=1)
    warped = depth.copy()
    warped[:, : w // 2] += 0.5
    bad = check_pose_depth(depth=warped, vertices=verts, cam_t=cam_t, focal_length=f, img_size=(w, h), subsample=1)
    assert good["ok"] and good["mae_m"] < 0.02, good
    assert bad["mae_m"] > 0.2, (good, bad)
    print(good, bad)
