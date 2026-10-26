"""TokenHMR worker for the app. Run inside conda env tkhmr.

Reads ipc/live.jpg, writes live_joints.npy and live_overlay.jpg (SMPL on RGB).
"""

from pathlib import Path
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)
os.environ["PYOPENGL_PLATFORM"] = "egl"

from lib.datasets.vitdet_dataset import ViTDetDataset
from lib.models import load_tokenhmr
from lib.utils import recursive_to
from lib.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
_OP_TO_HML = [8, 12, 9, 1, 13, 10, 1, 14, 11, 1, 14, 11, 1, 5, 2, 0, 5, 2, 6, 3, 7, 4]


def _box(h, w):
    m = 0.08
    return np.array([[m * w, m * h, (1 - m) * w, (1 - m) * h]], dtype=np.float32)


def _verts(model, pred_smpl):
    params = {k: v for k, v in pred_smpl.items()}
    b = params["betas"].shape[0]
    params["global_orient"] = params["global_orient"].reshape(b, -1, 3, 3)
    params["body_pose"] = params["body_pose"].reshape(b, -1, 3, 3)
    params["betas"] = params["betas"].reshape(b, -1)
    return model.smpl(**params, pose2rot=False).vertices[0].detach().cpu().numpy()


def _overlay(bgr, verts, cam_t, renderer, focal):
    h, w = bgr.shape[:2]
    rgba = renderer.render_rgba_multiple(
        [verts],
        cam_t=[cam_t.copy()],
        render_res=[w, h],
        focal_length=focal,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
    )
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    alpha = rgba[:, :, 3:]
    out = rgb * (1.0 - alpha) + rgba[:, :, :3] * alpha
    return (255.0 * out[:, :, ::-1]).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", type=Path, required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "data/checkpoints/tokenhmr_model_latest.ckpt"))
    parser.add_argument("--model_config", default=str(ROOT / "data/checkpoints/model_config.yaml"))
    args = parser.parse_args()
    ipc = args.ipc
    ipc.mkdir(parents=True, exist_ok=True)
    frame_path = ipc / "live.jpg"
    joints_path = ipc / "live_joints.npy"
    overlay_path = ipc / "live_overlay.jpg"
    ready = ipc / "worker.ready"
    stop = ipc / "worker.stop"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_tokenhmr(args.checkpoint, args.model_config, is_train_state=False, is_demo=True)
    model = model.to(device).eval()
    renderer = Renderer(cfg, faces=model.smpl.faces)
    ready.write_text("ok")
    last_mtime = -1.0

    while not stop.exists():
        if not frame_path.exists():
            time.sleep(0.03)
            continue
        mtime = frame_path.stat().st_mtime
        if mtime <= last_mtime:
            time.sleep(0.02)
            continue
        last_mtime = mtime
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        dataset = ViTDetDataset(cfg, bgr, _box(h, w))
        batch = recursive_to(next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0))), device)
        with torch.no_grad():
            out = model(batch)
            verts = _verts(model, out["pred_smpl_params"])
        k = out["pred_keypoints_3d"][0].detach().cpu().numpy()
        idx = [i if i < len(k) else 0 for i in _OP_TO_HML]
        np.save(joints_path, k[idx].astype(np.float32))
        focal = float(cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * batch["img_size"].float().max().detach().cpu())
        cam_t = cam_crop_to_full(
            out["pred_cam"], batch["box_center"].float(), batch["box_size"].float(),
            batch["img_size"].float(), focal,
        )[0].detach().cpu().numpy()
        vis = _overlay(bgr, verts, cam_t, renderer, focal)
        cv2.imwrite(str(overlay_path), vis)

    if ready.exists():
        ready.unlink()


if __name__ == "__main__":
    main()
