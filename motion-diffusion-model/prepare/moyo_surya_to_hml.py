#!/usr/bin/env python3
"""Turn the Surya Namaskar MOYO SMPL takes into MDM HumanML3D clips.

The source pkls are 60 fps and already Y-up. MDM trains on 20 fps vectors of
shape (frames, 263), each clip between 40 and 199 frames. This keeps 6 seconds
centered on the hold frame marked in the MOYO toolkit, taking every 3rd frame.
"""
import os
import sys

import numpy as np

# chumpy 0.70 still imports numpy.bool, which NumPy 1.24 removed.
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

import joblib
import torch
import smplx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.scripts import motion_process as mp
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets

MOYO = os.path.join(
    ROOT, "..", "3d_pose", "TokenHMR", "datasets", "moyo_toolkit", "data", "mosh_smpl"
)
SMPL_DIR = os.path.join(
    ROOT, "..", "3d_pose", "TokenHMR", "datasets", "data", "body_models"
)
OUT = os.path.join(ROOT, "dataset", "HumanML3D")

SRC_FPS = 60
DST_FPS = 20
STEP = SRC_FPS // DST_FPS
WIN_20 = 120  # 6 seconds at 20 fps. MDM accepts 40..199 frames.

# filename piece -> (clip id, 30fps image frame of the hold, caption, token line)
# Frame numbers are the 20220923 entries in moyo/utils/constants.py.
CLIPS = {
    "Cobra_Pose_or_Bhujangasana_-a": (
        "cobra_a", 278,
        "a person lies face down and lifts the chest into a yoga pose",
        "a/DET person/NOUN lie/VERB face/NOUN down/ADV and/OTHER lift/VERB the/DET chest/NOUN into/ADP a/DET yoga/NOUN pose/NOUN",
    ),
    "Cobra_Pose_or_Bhujangasana_-b": (
        "cobra_b", 294,
        "a person lies face down and lifts the chest into a yoga pose",
        "a/DET person/NOUN lie/VERB face/NOUN down/ADV and/OTHER lift/VERB the/DET chest/NOUN into/ADP a/DET yoga/NOUN pose/NOUN",
    ),
    "Cobra_Pose_or_Bhujangasana_-c": (
        "cobra_c", 551,
        "a person lies face down and lifts the chest into a yoga pose",
        "a/DET person/NOUN lie/VERB face/NOUN down/ADV and/OTHER lift/VERB the/DET chest/NOUN into/ADP a/DET yoga/NOUN pose/NOUN",
    ),
    "Cobra_Pose_or_Bhujangasana_-d": (
        "cobra_d", 281,
        "a person lies face down and lifts the chest into a yoga pose",
        "a/DET person/NOUN lie/VERB face/NOUN down/ADV and/OTHER lift/VERB the/DET chest/NOUN into/ADP a/DET yoga/NOUN pose/NOUN",
    ),
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a": (
        "downdog_a", 158,
        "a person holds a downward dog yoga pose with hips lifted and arms straight",
        "a/DET person/NOUN hold/VERB a/DET downward/ADV dog/NOUN yoga/NOUN pose/NOUN with/ADP hip/NOUN lift/VERB and/OTHER arm/NOUN straight/ADJ",
    ),
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-b": (
        "downdog_b", 257,
        "a person holds a downward dog yoga pose with hips lifted and arms straight",
        "a/DET person/NOUN hold/VERB a/DET downward/ADV dog/NOUN yoga/NOUN pose/NOUN with/ADP hip/NOUN lift/VERB and/OTHER arm/NOUN straight/ADJ",
    ),
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-c": (
        "downdog_c", 165,
        "a person holds a downward dog yoga pose with hips lifted and arms straight",
        "a/DET person/NOUN hold/VERB a/DET downward/ADV dog/NOUN yoga/NOUN pose/NOUN with/ADP hip/NOUN lift/VERB and/OTHER arm/NOUN straight/ADJ",
    ),
    "Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a": (
        "chaturanga_a", 197,
        "a person holds a low plank yoga pose with elbows bent close to the floor",
        "a/DET person/NOUN hold/VERB a/DET low/ADJ plank/NOUN yoga/NOUN pose/NOUN with/ADP elbow/NOUN bent/ADJ close/ADJ to/ADP the/DET floor/NOUN",
    ),
    "Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-b": (
        "chaturanga_b", 140,
        "a person holds a low plank yoga pose with elbows bent close to the floor",
        "a/DET person/NOUN hold/VERB a/DET low/ADJ plank/NOUN yoga/NOUN pose/NOUN with/ADP elbow/NOUN bent/ADJ close/ADJ to/ADP the/DET floor/NOUN",
    ),
    "Plank_Pose_or_Kumbhakasana_-a": (
        "plank_a", 231,
        "a person holds a straight yoga plank pose with arms straight",
        "a/DET person/NOUN hold/VERB a/DET straight/ADJ yoga/NOUN plank/NOUN pose/NOUN with/ADP arm/NOUN straight/ADJ",
    ),
    "Low_Lunge_pose_or_Anjaneyasana_-a": (
        "lunge_a", 545,
        "a person holds a low lunge yoga pose with one knee bent forward",
        "a/DET person/NOUN hold/VERB a/DET low/ADJ lunge/NOUN yoga/NOUN pose/NOUN with/ADP one/NUM knee/NOUN bent/ADJ forward/ADV",
    ),
    "Low_Lunge_pose_or_Anjaneyasana_-b": (
        "lunge_b", 432,
        "a person holds a low lunge yoga pose with one knee bent forward",
        "a/DET person/NOUN hold/VERB a/DET low/ADJ lunge/NOUN yoga/NOUN pose/NOUN with/ADP one/NUM knee/NOUN bent/ADJ forward/ADV",
    ),
    "Low_Lunge_pose_or_Anjaneyasana_-c": (
        "lunge_c", 260,
        "a person holds a low lunge yoga pose with one knee bent forward",
        "a/DET person/NOUN hold/VERB a/DET low/ADJ lunge/NOUN yoga/NOUN pose/NOUN with/ADP one/NUM knee/NOUN bent/ADJ forward/ADV",
    ),
    "Low_Lunge_pose_or_Anjaneyasana_-d": (
        "lunge_d", 380,
        "a person holds a low lunge yoga pose with one knee bent forward",
        "a/DET person/NOUN hold/VERB a/DET low/ADJ lunge/NOUN yoga/NOUN pose/NOUN with/ADP one/NUM knee/NOUN bent/ADJ forward/ADV",
    ),
    "Standing_Forward_Bend_pose_or_Uttanasana_-a": (
        "uttanasana_a", 168,
        "a person stands and bends forward in a yoga pose",
        "a/DET person/NOUN stand/VERB and/OTHER bend/VERB forward/ADV in/ADP a/DET yoga/NOUN pose/NOUN",
    ),
    "Standing_Forward_Bend_pose_or_Uttanasana_-b": (
        "uttanasana_b", 135,
        "a person stands and bends forward in a yoga pose",
        "a/DET person/NOUN stand/VERB and/OTHER bend/VERB forward/ADV in/ADP a/DET yoga/NOUN pose/NOUN",
    ),
}


def hold_window(n_frames, image_frame):
    """6 seconds centered on the 30 fps hold frame. Images are 30 fps, mocap is 60."""
    need = WIN_20 * STEP
    center = int(image_frame * (SRC_FPS / 30.0))
    start = center - need // 2
    start = max(0, min(start, max(0, n_frames - need)))
    return start, min(n_frames, start + need)


def joints_22(model, seq, start, end):
    idx = np.arange(start, end, STEP)
    betas = np.repeat(seq["betas"][idx[0], :10][None], len(idx), axis=0)
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(seq["global_orient"][idx]).float(),
            body_pose=torch.tensor(seq["body_pose"][idx]).float(),
            betas=torch.tensor(betas).float(),
            transl=torch.tensor(seq["transl"][idx]).float(),
        )
    return out.joints[:, :22].cpu().numpy()


def prepare_skeleton(example_joints):
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
    mp.tgt_offsets = skel.get_offsets_joints(torch.from_numpy(example_joints[0]).float())


def main():
    model = smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", num_betas=10)
    motion_dir = os.path.join(OUT, "new_joint_vecs")
    text_dir = os.path.join(OUT, "texts")
    os.makedirs(motion_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    found = {}
    for split in ("train", "val"):
        folder = os.path.join(MOYO, split)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            for key in CLIPS:
                if key in name and name.endswith(".pkl"):
                    found[key] = os.path.join(folder, name)

    missing = [k for k in CLIPS if k not in found]
    if missing:
        raise SystemExit("missing pkls:\n" + "\n".join(missing))

    first_key = next(iter(CLIPS))
    seq0 = joblib.load(found[first_key])
    s0, e0 = hold_window(seq0["transl"].shape[0], CLIPS[first_key][1])
    prepare_skeleton(joints_22(model, seq0, s0, e0))

    ids = []
    for key, (clip_id, image_frame, caption, tokens) in CLIPS.items():
        seq = joblib.load(found[key])
        n = seq["transl"].shape[0]
        start, end = hold_window(n, image_frame)
        joints = joints_22(model, seq, start, end)
        feats, _, _, _ = mp.process_file(joints, 0.002)
        if feats.shape[0] < 40 or feats.shape[0] >= 200 or feats.shape[1] != 263:
            raise SystemExit(f"{clip_id} bad shape {feats.shape}")
        np.save(os.path.join(motion_dir, clip_id + ".npy"), feats.astype(np.float32))
        with open(os.path.join(text_dir, clip_id + ".txt"), "w") as f:
            f.write(f"{caption}#{tokens}#0.0#0.0\n")
        ids.append(clip_id)
        print(f"{clip_id:16} frames {feats.shape[0]:3d}  pelvis_height {feats[0, 3]:.3f}  src {n} @60fps")

    val_ids = {"cobra_d", "downdog_c"}
    train_ids = [i for i in ids if i not in val_ids]
    with open(os.path.join(OUT, "train.txt"), "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(os.path.join(OUT, "val.txt"), "w") as f:
        f.write("\n".join(i for i in ids if i in val_ids) + "\n")
    # sample.generate always opens split 'test', even for a typed prompt.
    with open(os.path.join(OUT, "test.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"wrote {len(ids)} clips into {OUT}")
    print("left Mean.npy and Std.npy unchanged")


if __name__ == "__main__":
    main()
