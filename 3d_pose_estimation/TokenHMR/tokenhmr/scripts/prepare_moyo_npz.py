#!/usr/bin/env python3
"""Convert one MOYO take (images + mosh_smpl + cameras) to TokenHMR ImageDataset npz."""
import argparse
import glob
import json
import os
import sys

import cv2
import joblib
import numpy as np
import torch

# TokenHMR SMPL (OpenPose + extra joints)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.models.smpl_wrapper import SMPL
from lib.utils.geometry import aa_to_rotmat


def project(X, R, t, f, cx, cy):
    Xc = (R @ X.T).T + t
    z = np.clip(Xc[:, 2], 1e-6, None)
    u = f * Xc[:, 0] / z + cx
    v = f * Xc[:, 1] / z + cy
    return np.stack([u, v, Xc[:, 2]], axis=1), Xc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', required=True, help='Folder that contains the take directory')
    p.add_argument('--take', required=True, help='Take folder name')
    p.add_argument('--smpl_pkl', required=True)
    p.add_argument('--cameras_json', required=True)
    p.add_argument('--smpl_model', default='data/body_models/smpl')
    p.add_argument('--j_extra', default='data/body_models/SMPL_to_J19.pkl')
    p.add_argument('--out_npz', required=True)
    p.add_argument('--skip', type=int, default=5)
    p.add_argument('--cams', default='1,2,3,4,5,6,7,8')
    args = p.parse_args()

    seq = joblib.load(args.smpl_pkl)
    N = seq['transl'].shape[0]
    with open(args.cameras_json) as f:
        cameras = json.load(f)

    device = 'cpu'
    smpl = SMPL(
        model_path=args.smpl_model,
        gender='neutral',
        joint_regressor_extra=args.j_extra,
    ).to(device)

    want_cams = {int(c) for c in args.cams.split(',') if c.strip()}
    take_dir = os.path.join(args.img_dir, args.take)

    imgnames, centers, scales = [], [], []
    body_pose, betas, has_pose, has_betas = [], [], [], []
    kp2d, kp3d = [], []

    for cam_id in sorted(want_cams):
        cam_dir = os.path.join(take_dir, f'YOGI_Cam_{cam_id:02d}')
        paths = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
        if not paths:
            print(f'skip missing {cam_dir}')
            continue
        cam = cameras[f'cam_{cam_id}']
        R = np.asarray(cam['rotation'], dtype=np.float32)
        C = np.asarray(cam['position'], dtype=np.float32) / 1000.0  # mm -> m
        t = (-R @ C).astype(np.float32)
        f, (cx, cy) = float(cam['focal']), cam['princpt']

        for pi, path in enumerate(paths):
            if pi % args.skip:
                continue
            img_i = int(os.path.splitext(path)[0].split('_')[-1])
            mocap_i = int(np.clip(2 * (img_i - 1), 0, N - 1))

            go = aa_to_rotmat(torch.from_numpy(seq['global_orient'][mocap_i:mocap_i + 1]).float()).view(1, 1, 3, 3)
            bp = aa_to_rotmat(torch.from_numpy(seq['body_pose'][mocap_i:mocap_i + 1]).float().reshape(-1, 3)).view(1, 23, 3, 3)
            bt = torch.from_numpy(seq['betas'][mocap_i:mocap_i + 1, :10]).float()
            tr = torch.from_numpy(seq['transl'][mocap_i:mocap_i + 1]).float()
            out = smpl(global_orient=go, body_pose=bp, betas=bt, transl=tr, pose2rot=False)
            joints = out.joints[0].detach().cpu().numpy()  # (44, 3) OpenPose+extra

            pj, Xc = project(joints, R, t, f, cx, cy)
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            vis = (pj[:, 0] >= 0) & (pj[:, 0] < w) & (pj[:, 1] >= 0) & (pj[:, 1] < h) & (pj[:, 2] > 0.1)
            if vis.sum() < 8:
                continue

            xs, ys = pj[vis, 0], pj[vis, 1]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            bw, bh = (x2 - x1) * 1.2, (y2 - y1) * 1.2
            cx_b, cy_b = (x1 + x2) / 2, (y1 + y2) / 2
            scale = float(max(bw, bh))

            body25 = np.zeros((25, 3), np.float32)
            extra19 = np.zeros((19, 3), np.float32)
            n_op = min(25, pj.shape[0])
            body25[:n_op, :2] = pj[:n_op, :2]
            body25[:n_op, 2] = vis[:n_op].astype(np.float32)
            if pj.shape[0] > 25:
                n_ex = min(19, pj.shape[0] - 25)
                extra19[:n_ex, :2] = pj[25:25 + n_ex, :2]
                extra19[:n_ex, 2] = vis[25:25 + n_ex].astype(np.float32)

            # pelvis-rooted camera-frame 3D (OpenPose index 8 ~ pelvis)
            pelvis = Xc[8] if vis[8] else Xc[vis].mean(0)
            j3 = np.zeros((25, 4), np.float32)
            j3[:n_op, :3] = Xc[:n_op] - pelvis
            j3[:n_op, 3] = vis[:n_op].astype(np.float32)
            e3 = np.zeros((19, 4), np.float32)
            if pj.shape[0] > 25:
                e3[:n_ex, :3] = Xc[25:25 + n_ex] - pelvis
                e3[:n_ex, 3] = vis[25:25 + n_ex].astype(np.float32)

            imgnames.append(os.path.relpath(path, args.img_dir))
            centers.append([cx_b, cy_b])
            scales.append([scale, scale])
            body_pose.append(np.concatenate([seq['global_orient'][mocap_i], seq['body_pose'][mocap_i]]))
            betas.append(seq['betas'][mocap_i, :10])
            has_pose.append(1.0)
            has_betas.append(1.0)
            kp2d.append(np.concatenate([body25, extra19], 0))
            kp3d.append(np.concatenate([j3, e3], 0))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez(
        args.out_npz,
        imgname=np.array(imgnames),
        center=np.asarray(centers, np.float32),
        scale=np.asarray(scales, np.float32),
        body_pose=np.asarray(body_pose, np.float32),
        betas=np.asarray(betas, np.float32),
        has_body_pose=np.asarray(has_pose, np.float32),
        has_betas=np.asarray(has_betas, np.float32),
        body_keypoints_2d=np.asarray(kp2d, np.float32)[:, :25],
        extra_keypoints_2d=np.asarray(kp2d, np.float32)[:, 25:],
        body_keypoints_3d=np.asarray(kp3d, np.float32)[:, :25],
        extra_keypoints_3d=np.asarray(kp3d, np.float32)[:, 25:],
    )
    print(f'wrote {args.out_npz}  n={len(imgnames)}')


if __name__ == '__main__':
    main()
