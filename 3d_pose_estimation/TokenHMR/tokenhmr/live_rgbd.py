"""Realtime RealSense RGB-D → TokenHMR mesh overlay + depth check.

Same 3D body overlay as tokenhmr/demo.py --full_frame. The mesh pose only
updates when MotionGate (from track.py) sees motion vs the locked crop.

Run from the TokenHMR repo root:

  python tokenhmr/live_rgbd.py \\
      --checkpoint data/checkpoints/tokenhmr_model_latest.ckpt \\
      --model_config data/checkpoints/model_config.yaml

q quits. --detectron uses the demo person detector. --webcam 0 skips RealSense.
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

from depth_check import check_pose_depth
from lib.datasets.vitdet_dataset import ViTDetDataset
from lib.models import load_tokenhmr
from lib.utils import recursive_to
from lib.utils.renderer import Renderer, cam_crop_to_full
from motion_gate import MotionGate, MotionGateConfig

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def _default_box(h, w, kps=None):
    if kps is not None and len(kps) > 0:
        x0, y0 = kps.min(axis=0)
        x1, y1 = kps.max(axis=0)
        pad = 0.2 * max(x1 - x0, y1 - y0, 80)
        return np.array([[x0 - pad, y0 - pad, x1 + pad, y1 + pad]], dtype=np.float32)
    m = 0.08
    return np.array([[m * w, m * h, (1 - m) * w, (1 - m) * h]], dtype=np.float32)


def _clip_box(box, h, w):
    box = box.copy()
    box[:, 0] = np.clip(box[:, 0], 0, w - 2)
    box[:, 1] = np.clip(box[:, 1], 0, h - 2)
    box[:, 2] = np.clip(box[:, 2], box[:, 0] + 2, w - 1)
    box[:, 3] = np.clip(box[:, 3], box[:, 1] + 2, h - 1)
    return box


def _open_realsense(width, height, fps):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def read():
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
        return color, depth

    return read, pipeline.stop


def _open_webcam(index, width, height, fps):
    cap = cv2.VideoCapture(int(index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"could not open webcam {index}")

    def read():
        ok, frame = cap.read()
        if not ok:
            return None, None
        return frame, None

    return read, cap.release


def _gated_mesh(model, pred_smpl):
    params = {k: v for k, v in pred_smpl.items()}
    b = params["betas"].shape[0]
    params["global_orient"] = params["global_orient"].reshape(b, -1, 3, 3)
    params["body_pose"] = params["body_pose"].reshape(b, -1, 3, 3)
    params["betas"] = params["betas"].reshape(b, -1)
    smpl_out = model.smpl(**params, pose2rot=False)
    return smpl_out.vertices[0].detach().cpu().numpy(), smpl_out.joints[0].detach().cpu().numpy()


def _blend_mesh(bgr, rgba):
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    alpha = rgba[:, :, 3:]
    out = rgb * (1.0 - alpha) + rgba[:, :, :3] * alpha
    return (255.0 * out[:, :, ::-1]).astype(np.uint8)


def _overlay_mesh(bgr, verts, cam_t, renderer, focal):
    h, w = bgr.shape[:2]
    rgba = renderer.render_rgba_multiple(
        [verts],
        cam_t=[cam_t.copy()],
        render_res=[w, h],
        focal_length=focal,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
    )
    return _blend_mesh(bgr, rgba), rgba


def _draw_hud(img, fps, depth_report, gate, unlocked):
    img_m = pose_m = None
    if gate.last_img_motion is not None:
        img_m = float(gate.last_img_motion[0])
        pose_m = float(gate.last_pose_motion[0])
    gate_txt = "mesh OPEN" if unlocked else "mesh HOLD"
    lines = [f"{fps:.1f} fps  {gate_txt}"]
    if img_m is not None:
        lines.append(f"motion img {img_m:.3f}/{gate.pixel_thresh}  pose {pose_m:.1f}/{gate.pose_thresh_deg} deg")
    if depth_report is None:
        lines.append("depth: none")
    else:
        mae = depth_report.get("mae_m")
        tag = "OK" if depth_report.get("ok") else "WEAK"
        extra = f"mae {mae:.2f}m" if mae is not None else depth_report.get("reason", "")
        lines.append(f"depth {tag}  score {depth_report.get('score')}  {extra}")
    y = 24
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2, cv2.LINE_AA)
        y += 22
    return img


def main():
    here = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(here / "data/checkpoints/tokenhmr_model_latest.ckpt"))
    parser.add_argument("--model_config", default=str(here / "data/checkpoints/model_config.yaml"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--webcam", type=int, default=-1, help="OpenCV camera index instead of RealSense")
    parser.add_argument("--detectron", action="store_true", help="run the demo person detector")
    parser.add_argument("--detect-every", type=int, default=15)
    parser.add_argument("--pixel-thresh", type=float, default=0.05)
    parser.add_argument("--pose-thresh-deg", type=float, default=12.0)
    parser.add_argument("--unlock-frames", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_tokenhmr(
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_config,
        is_train_state=False,
        is_demo=True,
    )
    model = model.to(device).eval()
    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    gate = MotionGate(MotionGateConfig(
        pixel_thresh=args.pixel_thresh,
        pose_thresh_deg=args.pose_thresh_deg,
        unlock_frames=args.unlock_frames,
    ))

    detector = None
    if args.detectron:
        from detectron2.config import LazyConfig
        from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
        import lib

        cfg_path = Path(lib.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        det_cfg = LazyConfig.load(str(cfg_path))
        det_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(det_cfg)

    if args.webcam >= 0:
        read, close = _open_webcam(args.webcam, args.width, args.height, args.fps)
        print("webcam RGB only, depth check off")
    else:
        try:
            read, close = _open_realsense(args.width, args.height, args.fps)
        except Exception as exc:
            raise SystemExit(
                f"RealSense failed ({exc}). Install pyrealsense2 or pass --webcam 0"
            ) from exc

    last_kps = None
    last_rgba = None
    last_cam_t = None
    last_verts = None
    last_focal = None
    frame_i = 0
    t0 = time.time()
    shown = 0
    try:
        while True:
            bgr, depth = read()
            if bgr is None:
                break
            h, w = bgr.shape[:2]

            boxes = None
            if detector is not None and (frame_i % max(1, args.detect_every) == 0 or last_kps is None):
                det = detector(bgr)["instances"]
                keep = (det.pred_classes == 0) & (det.scores > 0.5)
                boxes = det.pred_boxes.tensor[keep].cpu().numpy()
            if boxes is None or len(boxes) == 0:
                boxes = _clip_box(_default_box(h, w, last_kps), h, w)
            elif len(boxes) > 1:
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes[area.argmax() : area.argmax() + 1]

            dataset = ViTDetDataset(model_cfg, bgr, boxes)
            batch = recursive_to(next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0))), device)
            with torch.no_grad():
                out = model(batch)
                img = batch["img"]
                mask = torch.ones(img.shape[0], 1, img.shape[2], img.shape[3], device=img.device, dtype=img.dtype)
                x = torch.cat([img, mask], dim=1)
                unlocked = bool(gate.apply(out["pred_smpl_params"], out["pred_cam"], x)[0])
                verts, _ = _gated_mesh(model, out["pred_smpl_params"])

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal = float(
                (model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()).detach().cpu()
            )
            cam_t = cam_crop_to_full(out["pred_cam"], box_center, box_size, img_size, scaled_focal)[0].detach().cpu().numpy()

            if unlocked or last_rgba is None:
                vis, last_rgba = _overlay_mesh(bgr, verts, cam_t, renderer, scaled_focal)
                last_cam_t, last_verts, last_focal = cam_t, verts, scaled_focal
            else:
                vis = _blend_mesh(bgr, last_rgba)

            x0, y0, x1, y1 = boxes[0]
            last_kps = np.array([[x0, y0], [x1, y1]], dtype=np.float32)

            report = None
            if depth is not None:
                report = check_pose_depth(
                    depth=depth,
                    vertices=last_verts,
                    cam_t=last_cam_t,
                    focal_length=last_focal,
                    img_size=(w, h),
                )
                dshow = np.clip(depth / 4.0, 0, 1)
                dcolor = cv2.applyColorMap((255 * dshow).astype(np.uint8), cv2.COLORMAP_JET)
                small = cv2.resize(dcolor, (160, 120))
                vis[8:128, w - 168 : w - 8] = small

            shown += 1
            fps = shown / max(1e-6, time.time() - t0)
            vis = _draw_hud(vis, fps, report, gate, unlocked)
            cv2.imshow("TokenHMR RGB-D", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_i += 1
    finally:
        close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
