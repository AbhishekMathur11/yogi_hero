"""Bake an MDM SMPL mesh into a looping body video. Run in the mdm env."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _load_smpl(path):
    data = np.load(path, allow_pickle=True)
    pack = data.item() if hasattr(data, "item") else data
    if isinstance(pack, np.ndarray):
        pack = pack.tolist()
    verts = pack["vertices"]
    if hasattr(verts, "detach"):
        verts = verts.detach().cpu().numpy()
    verts = np.asarray(verts)
    faces = np.asarray(pack["faces"])
    if verts.ndim == 3 and verts.shape[1] == 3:
        pass  # (V, 3, T)
    elif verts.ndim == 3 and verts.shape[2] == 3:
        verts = np.transpose(verts, (1, 2, 0))
    else:
        raise ValueError(f"unexpected vertices shape {verts.shape}")
    return verts.astype(np.float32), faces.astype(np.int32)


def bake(smpl_path, out_mp4, width=640, height=480, fps=20):
    import pyrender
    import trimesh

    verts, faces = _load_smpl(smpl_path)
    v, _, tcount = verts.shape
    center = verts.reshape(v, 3, tcount).mean(axis=(0, 2))
    renderer = pyrender.OffscreenRenderer(width, height)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    pose = np.eye(4)
    pose[1, 3] = 0.15
    pose[2, 3] = 2.4
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    color = (0.82, 0.72, 0.58, 1.0)
    mat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.05, roughnessFactor=0.7, baseColorFactor=color)
    for i in range(tcount):
        mesh = trimesh.Trimesh(vertices=verts[:, :, i] - center, faces=faces, process=False)
        scene = pyrender.Scene(bg_color=[12, 11, 10, 255], ambient_light=[0.35, 0.32, 0.28])
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat))
        scene.add(cam, pose=pose)
        scene.add(light, pose=pose)
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        writer.write(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
    writer.release()
    renderer.delete()
    return out_mp4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smpl", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bake(args.smpl, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
