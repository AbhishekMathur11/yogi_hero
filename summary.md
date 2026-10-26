# Yoga coach codebase (`yoga_tr`)

This repo is an AI yoga coach for the Nebius NVIDIA AI hackathon (Physical AI). The product is a **Surya Namaskar (sun salutation) coach**: show a pose, watch the person copy it, score the body with joint-angle math, speak one correction, then advance when the pose is held well enough.

Hardware in mind: Intel RealSense RGB-D, local RTX 5070 (12 GB). The stack must include an NVIDIA model and Nebius Token Factory / AI Cloud somewhere. As of this writing, **Nemotron on Nebius is still a placeholder**. The motion model in the live design is **MDM** (Guy Tevet), fine-tuned on a handful of MOYO yoga clips. NVIDIA models that were tried (Kimodo, GEM) live under `extras/` and are **not** in the live loop.

There is **no single app** that runs the camera-to-coach loop yet. The folders are components. Glue (live capture, UI overlay, stored reference poses per asana, real Nemotron calls) is still to be built.

---

## Intended product loop

Do not reopen this unless the product itself changes.

```
DEMO  →  IMITATE  →  EVALUATE  →  COACH  →  ADVANCE
```

1. **DEMO.** Show the current asana from `guidance_agent/plan.json`. The motion model (MDM) is for a **reference overlay / ghost**, not for scoring. Pre-generate clips offline. Do not sample MDM every camera frame; 50-step diffusion is too slow for realtime.
2. **IMITATE.** TokenHMR reads the person from RGB (RealSense). Output is SMPL, then **22 joints × 3 XYZ** (Y up, HumanML3D / SMPL order). Depth/IMU was planned as a check; `check_pose_depth` is still a stub and returns `None`.
3. **EVALUATE.** Local math in `feedback_agent/feedback.py`. About 11 angles (knees, hips, elbows, shoulders, ankles) plus spine lean. Result is a JSON-like dict: score in `[0, 1]`, per-joint errors, worst joint. Not a VLM.
4. **COACH.** One spoken correction. `nemotron_coach` / `nemotron_line` are placeholders. The real call should send that JSON to **Nemotron on Nebius Token Factory** and never block the camera thread.
5. **ADVANCE.** Guidance advances when score stays **≥ 0.85 for ~1.5 seconds**. A dip resets the hold. Next asana name comes from `plan.json`, not from the LLM inventing a sequence. Optional later: Nemotron may **rename** the next asana if the person fails twice; it still does not invent the whole flow.

Scoring is **streaming while they adjust**, not a single frozen frame.

Wire the two agents like this:

```python
from feedback_agent.feedback import compare
from guidance_agent.guidance import Guidance

guide = Guidance()
report = compare(live_joints, reference_joints)  # both (22, 3)
state = guide.tick(report["score"])
# state["asana"]  → what to show
# report["coach"]["text"] → one correction
```

---

## Top-level layout

```
yoga_tr/
  3d_pose_estimation/TokenHMR/   live 3D pose from images (not yet a camera loop)
  extras/
    kimodo/                      NVIDIA text-to-motion (tried, not in live loop)
    GENMO/                       NVIDIA GEM motion (out of live loop)
    data_river/moyo_toolkit/     MOYO download / constants (not a converter)
  feedback_agent/                joint-angle scorer + coach placeholder
  guidance_agent/                plan.json state machine + speech placeholder
  motion-diffusion-model/        MDM: yoga fine-tune, sample, stick/mesh viz
  summary.md                     this file
```

There is **no** `stmc/` folder. STMC weights on Google Drive 404'd; MDM was chosen instead.

Older notes may say `3d_pose/TokenHMR`. The real tree is **`3d_pose_estimation/TokenHMR`**. MDM’s converter and the `SMPL_NEUTRAL.pkl` symlink were first pointed at the old path; the symlink was later retargeted. The converter still has the old `../3d_pose/TokenHMR/...` strings in it.

---

## `guidance_agent/`

Small custom code. Two files.

| File | Role |
|------|------|
| `plan.json` | Ordered asana **names**. Sequence is a simplified sun salutation: prayer → uttanasana → lunge → plank → chaturanga → cobra → downdog → lunge → uttanasana → prayer. |
| `guidance.py` | `Guidance` class. Phases: `demo` → `imitate` → (`demo` of next) … → `done`. |

Rules:

- Does **not** invent the next pose.
- `begin_imitate()` flips `demo` → `imitate`.
- `tick(score)` also leaves `demo` if you start sending scores.
- Hold: score ≥ `HOLD_SCORE` (0.85) for `HOLD_SECONDS` (1.5). Below 0.85 clears `_hold_since`.
- `nemotron_line(phase, asana, nxt)` is the Nebius placeholder. It only **names** the pose already in the plan.

**Missing for a real demo:** a map from each name in `plan.json` to a stored 22-joint (or SMPL) **reference pose**. Prayer / Tadasana is in the plan but **not** in the MOYO SMPL set used for MDM.

---

## `feedback_agent/`

One file: `feedback.py`.

Input: `live` and `target`, each shape `(22, 3)`, Y up, HumanML3D joint order:

```
0 pelvis, 1 left_hip, 2 right_hip, 3 spine1, 4 left_knee, 5 right_knee,
6 spine2, 7 left_ankle, 8 right_ankle, 9 spine3, 10 left_foot, 11 right_foot,
12 neck, 13 left_collar, 14 right_collar, 15 head,
16 left_shoulder, 17 right_shoulder, 18 left_elbow, 19 right_elbow,
20 left_wrist, 21 right_wrist
```

Measured angles (vertex in the middle):

- knees, hips, elbows, shoulders, ankles
- `spine`: degrees the chest (`spine3` − `pelvis`) leans away from +Y

`compare(live, target, tolerance_deg=25)`:

- Per joint, `error_deg = |live − target|`
- Piece score = `max(0, 1 − error / 25)`
- Overall score = mean of pieces
- `hold_ready` if score ≥ 0.85
- `worst` is the largest error
- `coach` comes from `nemotron_coach` (local template: straighten / bend that joint, or “Hold this pose.”)

Replace `nemotron_coach` with Nebius. Do not put the LLM on the camera thread.

This module does **not** talk to TokenHMR or MDM.

---

## `3d_pose_estimation/TokenHMR/`

Upstream: [TokenHMR](https://tokenhmr.is.tue.mpg.de) (CVPR 2024). Image → SMPL body (mesh + pose). Local, not fine-tuned for the live coach unless you run the extra script.

### What the model outputs

SMPL parameters (`global_orient`, `body_pose`, `betas`, camera). Joints can be taken from the SMPL FK; the first **22** joints match HumanML3D order and are what the scorer wants.

### Scripts that matter here

| Path | Role |
|------|------|
| `tokenhmr/demo.py` | Images in a folder → detect person (Detectron2 ViTDet) → TokenHMR → overlay / optional mesh. **Not a RealSense loop.** |
| `tokenhmr/track.py` | Video tracking via PHALP + TokenHMR. Still not the product camera loop. |
| `tokenhmr/depth_check.py` | **Stub.** `check_pose_depth(...)` returns `None` if no depth, and also returns `None` if depth is passed. Do not treat depth as implemented. |
| `tokenhmr/finetune_moyo.py` | Short fine-tune on a local MOYO npz. Optional, not required for the agreed live design (stock TokenHMR + depth check). |
| `tokenhmr/scripts/prepare_moyo_npz.py` | Images + mosh_smpl + cameras → TokenHMR ImageDataset npz. |
| `tokenhmr/lib/models/tokenhmr.py` | Model. |
| `datasets/data/body_models/smpl/SMPL_{NEUTRAL,FEMALE,MALE}.pkl` | Official SMPL pickles (need chumpy to unpickle). |
| `datasets/moyo_toolkit/data/mosh_smpl/{train,val}` | **Already-downloaded** MOYO SMPL fits used for MDM data. Keys: `betas (T, 300)`, `global_orient (T, 3)`, `body_pose (T, 69)`, `transl (T, 3)`. No fps field. Vicon/mocap is **60 fps**. Images in the full dataset are 30 fps (map `mocap_index ≈ 2 * image_frame`). |

Do **not** re-download the 741 GB image pack unless you actually need images for TokenHMR fine-tune. SMPL pkls are enough for MDM conversion.

### Still missing for realtime

A process that: RealSense RGB → TokenHMR → `(22, 3)` every frame, optionally depth, same joint convention as `feedback.py`.

---

## `motion-diffusion-model/`

Upstream: [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model) (MDM). Text-conditioned diffusion that generates **human motion as joint trajectories**, not a skinned avatar by default.

### What MDM actually outputs

Internally (HumanML3D / `--dataset humanml`):

- Per frame: **263-D** feature (`hml_vec`), **20 fps**, 22 joints.
- Loader skips clips with length **&lt; 40 or ≥ 200** frames (2.0–9.95 s). Max model length 196.

After `sample/generate.py`:

- `results.npy` is a dict:
  - `motion`: shape `(N, 22, 3, T)` — XYZ joint **positions**, Y up. Typical eval clip: `(1, 22, 3, 120)` = 6 s × 20 fps.
  - `text`, `lengths`, `num_samples`, `num_repetitions`
- Stick video: `samples_00_to_00.mp4` (the per-clip name `sample00_rep00.mp4` is often **not** written; `save_multiple_samples` writes the combined name).
- **No SMPL rotations and no mesh** in `results.npy`.

The 263-D vector is un-normalized (`inv_transform` with HumanML3D `Mean.npy` / `Std.npy`) then `recover_from_ric`. For this dataset `rot2xyz` is called with `pose_rep='xyz'` and **returns the joints unchanged**.

### Why `--dataset humanml` for yoga

That flag only selects the **HumanML3D loader** (263-D + CLIP text + Mean/Std). The original HumanML3D **corpus was never downloaded**. Yoga clips were written **into** `dataset/HumanML3D/` so the trainer would accept them. Fine-tuning is “yoga motions in HumanML3D format,” not training on walking/dancing HumanML3D.

### Yoga data conversion

`prepare/moyo_surya_to_hml.py` (custom):

1. Load 16 MOYO `mosh_smpl` pkls (all date **220923**).
2. SMPL forward kinematics, **10 betas**.
3. 6 s window centered on the toolkit **30 fps hold frame** (`moyo/utils/constants.py`, `frame_select_dict_combined`). Mocap index = `image_frame * 2`.
4. Keep every 3rd frame (60 → 20 fps).
5. `process_file(..., feet_thre=0.002)` → `(119, 263)` float32 (`input_frames − 1`).
6. Write `new_joint_vecs/{id}.npy` and `texts/{id}.txt` as `caption#tokens#0.0#0.0`.

Source pkls are **already Y-up**. Do **not** apply HumanML3D’s AMASS Z-up `trans_matrix`.

**Do not overwrite** `dataset/HumanML3D/Mean.npy` and `Std.npy`. They must stay original HumanML3D stats, shape `(263,)`.

Clips (16 total; train 14, val `cobra_d` + `downdog_c`; `test.txt` = all 16 because generate always opens the **test** split):

| id | Asana | Training caption (reuse this string when sampling) |
|----|--------|------------------------------------------------------|
| cobra_a..d | Bhujangasana | `a person lies face down and lifts the chest into a yoga pose` |
| downdog_a..c | Adho Mukha Svanasana | `a person holds a downward dog yoga pose with hips lifted and arms straight` |
| chaturanga_a,b | Chaturanga | `a person holds a low plank yoga pose with elbows bent close to the floor` |
| plank_a | Kumbhakasana | `a person holds a straight yoga plank pose with arms straight` |
| lunge_a..d | Anjaneyasana (lunge stand-in) | `a person holds a low lunge yoga pose with one knee bent forward` |
| uttanasana_a,b | Uttanasana | `a person stands and bends forward in a yoga pose` |

**Not in MOYO SMPL (cannot download them into this set):** Pranamasana / prayer, Hasta Uttanasana, Tadasana, Urdhva Mukha Svanasana (use cobra), Ashtanga Namaskara (use chaturanga), a joined sun-salutation sequence. Staff / side plank / warriors were left out on purpose.

Hold frames used (30 fps image index): cobra 278/294/551/281; downdog 158/257/165; chaturanga 197/140; plank 231; lunge 545/432/260/380; uttanasana 168/135.

GloVe: `glove/our_vab_{data.npy,idx.pkl,words.pkl}` from Hugging Face `guytevet/CLoSD` (Drive script is dead). Vocab 4199 × 300-d, includes `unk`. Words like cobra/warrior are UNK; yoga/plank/lunge/dog/downward are in vocab. Training uses **pre-tokenized** text; `en_core_web_sm` is not required.

### Fine-tune that actually ran

Conda env: **`mdm`**, Python 3.10.

```bash
conda activate mdm
cd ~/projects/yoga_tr/motion-diffusion-model
python -m train.train_mdm \
  --save_dir save/surya \
  --dataset humanml \
  --resume_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
  --diffusion_steps 50 \
  --use_ema \
  --batch_size 2 \
  --lr 1e-5 \
  --save_interval 500 \
  --num_steps 20000 \
  --train_platform_type NoPlatform
```

- Resume started at step **750000**. Last checkpoint **`save/surya/model000770007.pt`** (also `model000770000.pt`). There is no validation-picked “best.”
- `--diffusion_steps 50` must match the 50-step encoder. `train_args()` does **not** overwrite `diffusion_steps` from `args.json`; sampling **does** via `parse_and_load_from_model`.
- `--save_dir` must not already exist unless `--overwrite`.
- `--eval_during_training` is a bad idea on 14 clips.

### Sampling

```bash
conda activate mdm
cd ~/projects/yoga_tr/motion-diffusion-model
python -m sample.generate \
  --model_path ./save/surya/model000770007.pt \
  --text_prompt "a person lies face down and lifts the chest into a yoga pose" \
  --motion_length 6 \
  --num_samples 1 \
  --num_repetitions 1
```

`generate.py` hardcodes `split='test'` even with `--text_prompt`, so `dataset/HumanML3D/test.txt` must exist. `HumanML3D.__init__` asserts `len(dataset) > 1`. Prompts that were **not** training captions (e.g. prayer / “hands in prayer”) will still run but are not meaningful.

Quality caveat: the cobra sample’s 22 joints are a **small upright / inverted stick**, not a clear Bhujangasana (head Y below feet in the npy). 14-clip overfit on a locomotion-pretrained MDM. Do not claim the model “learned cobra.”

### Mesh / avatar from MDM (offline)

Stick mp4 is matplotlib bones. For a human mesh, MDM’s own SMPLify path:

```bash
python -m visualize.render_mesh \
  --input_path save/surya/<sample_folder>/samples_00_to_00.mp4
```

Behavior (after local edits):

- Parses `sample00_rep00.mp4` **or** `samples_00_to_00.mp4` (first index, rep 0).
- Reads `results.npy` next to the mp4 (the video file itself is unused).
- Writes `output_mesh/obj/frameXXX.obj` and `output_mesh/smpl_params.npy`.
- Fits SMPL to the 22 joints (`visualize/simplify_loc2rot.py`, ~150 iters). Slow. Needs `body_models/smpl/SMPL_NEUTRAL.pkl` and `visualize/joints2smpl/smpl_models/neutral_smpl_mean_params.h5`.

`SMPL_NEUTRAL.pkl` under MDM is a **symlink** to TokenHMR’s pickle. It must point at:

`3d_pose_estimation/TokenHMR/datasets/data/body_models/smpl/SMPL_NEUTRAL.pkl`

not the old `3d_pose/TokenHMR/...` path. Also needed: `body_models/smpl/J_regressor_extra.npy` shape `(9, 6890)` (from SPIN `data.tar.gz`). MDM constructs SMPL even for 263-D training because `Rotation2xyz` loads it.

This is **not** Kimodo’s Viser avatar. Kimodo stores joint **rotations** + skin. MDM stores joint **positions**; SMPLify recovers rotations after the fact.

### Local patches in this MDM clone (not upstream)

- `data_loaders/humanml/utils/plot_script.py`: Matplotlib 3.8 needs `fig.add_subplot(111, projection='3d')`. Old `Axes3D(fig)` produced title-only “empty” videos. Limits reset after `ax.clear()`.
- Numpy 1.24: `np.float` / `np.int` replaced with `float64` / `int64` in a few HumanML scripts.
- `prepare/moyo_surya_to_hml.py` and `test.txt` as above.
- `visualize/render_mesh.py`: combined mp4 name + `output_mesh/` output dir.

### Env gotchas (`conda` env `mdm`)

- GPU: RTX 5070 needs **torch with sm_120** (worked: `torch 2.14.1+cu130`). Torch 2.4 + cu121 → “no kernel image.”
- **numpy 1.24.1** + chumpy 0.70: patch `np.bool = np.bool_` etc. in chumpy `__init__.py` (wiped on reinstall) and in the converter. Install chumpy with `pip install --no-build-isolation chumpy`.
- **setuptools 81.x** (`>=77,<82`) so CLIP still has `pkg_resources`. 84 dropped it.
- `save/` is gitignored; checkpoints are large (~12 GB under `save/surya`). Do not delete unless asked.

---

## `extras/`

Not on the product critical path. Kept because they were evaluated or supply data helpers.

### `extras/kimodo/`

NVIDIA Kimodo: kinematic motion diffusion, Viser 3D demo, SMPL-X / SOMA / G1. Tried for yoga; **quality on cobra / Bhujangasana was poor**, and the public repo had no training code to teach unseen yoga. Warrior II generation worked better. **Out of the live loop.** Useful as a reference for “what a nice 3D avatar looks like” (rotations + skin in Viser). Do not drop MDM `results.npy` into Kimodo; formats differ.

### `extras/GENMO/` (repo now called GEM)

NVIDIA GEM: generalist motion (video / keypoints / text / audio). Explicitly **out of the live loop.**

### `extras/data_river/moyo_toolkit/`

Official MOYO toolkit: download scripts, `moyo/utils/constants.py` (hold frames), eval. **Not** a HumanML converter. `VICON_FPS = 60`. Download script can pull huge image zips; SMPL fits are the small pack. `smpl_fits.txt` lists URLs. Conversion for MDM lives in `motion-diffusion-model/prepare/moyo_surya_to_hml.py`.

---

## Data formats (cheat sheet)

| Thing | Shape / format | Where |
|--------|----------------|--------|
| Live / score pose | `(22, 3)` float, Y up | TokenHMR SMPL joints `[:22]`, `feedback.compare` |
| MDM training clip | `(~119, 263)` float32 | `dataset/HumanML3D/new_joint_vecs/*.npy` |
| MDM sample | `motion` `(N, 22, 3, T)` | `results.npy` |
| Text line | `caption#tokens#0.0#0.0` | `dataset/HumanML3D/texts/*.txt` |
| Mean / Std | `(263,)` original HumanML3D | `dataset/HumanML3D/Mean.npy`, `Std.npy` |
| MOYO SMPL pkl | `global_orient (T,3)`, `body_pose (T,69)`, `transl (T,3)`, `betas (T,300)` | TokenHMR `datasets/moyo_toolkit/data/mosh_smpl` |
| Sequence | JSON list of names | `guidance_agent/plan.json` |
| Mesh after SMPLify | `frameXXX.obj` + `smpl_params.npy` | sample folder `output_mesh/` |

---

## What is done vs what is not

**Done**

- Product loop agreed (DEMO / IMITATE / EVALUATE / COACH / ADVANCE).
- Tiny guidance + feedback agents with Nemotron stubs.
- MOYO → HumanML3D yoga set (16 clips) and MDM fine-tune from the 50-step HumanML encoder.
- Sampling + stick visualization (matplotlib 3.8 fix).
- Path to SMPL meshes from `results.npy` (SMPLify, offline).
- TokenHMR repo + SMPL bodies + MOYO mosh_smpl on disk.
- Kimodo / GEM / STMC explored; MDM chosen for the ghost.

**Not done (needed for a realtime demo)**

1. RealSense (or webcam) → TokenHMR → `(22, 3)` stream.
2. Depth check (or drop it).
3. Stored **reference pose per asana name** in `plan.json`.
4. One process: demo overlay, score, hold timer, advance.
5. UI: camera + ghost (play **precomputed** MDM / mesh, do not diffuse live).
6. Real **Nemotron on Nebius** for the one coach sentence (hackathon NVIDIA/Nebius requirement).
7. Optional TTS. Optional “failed twice → swap next name.”

**Known weak spots**

- MDM yoga samples are not anatomically convincing; overlay quality may need better references (raw MOYO SMPL FK is cleaner than sampled MDM).
- Converter / some comments still mention `3d_pose/TokenHMR`.
- Prayer pose is in the plan but not in the yoga MDM set.
- No UI, no live loop, no tests beyond `if __name__ == "__main__"` on the two agents.

---

## Hackathon / NVIDIA note

Stock MDM is **not** an NVIDIA model. The required NVIDIA piece in the current design is **Nemotron via Nebius Token Factory** (coach + optional next-name wording). Kimodo or GEM would also count if they were in the loop; they are not. Confirm exact Nemotron model IDs in the Token Factory console. Nemotron Nano VL is optional and not needed if scoring stays local math.
