# yogi_hero

Desktop yoga coach for the Nebius × NVIDIA Physical AI hackathon.

Left pane: 3D guide body from a motion diffusion model.  
Right pane: you (RealSense or webcam) with TokenHMR overlay.  
Bottom: joint-angle score + Nemotron coach on Nebius Token Factory.

The lesson is a **catalog**, not a hardcoded flow. Nemotron may suggest retry / skip / easier / next. Python only accepts asana ids that exist in `guidance_agent/asanas.json` and have a usable 3D clip.

## Run

Two conda envs stay separate (`mdm` for the app + MDM, `tkhmr` for pose).

```bash
conda activate mdm
cd /path/to/yogi_hero
# put the Nebius token in feedback_agent/api.yaml (one line, never commit)
python app/main.py
# optional: python app/main.py --webcam 0
```

## Add an asana later

1. Train or generate a 3D `body.mp4` for that pose.  
2. Add a row in `guidance_agent/asanas.json` with `plausible: true` and `in_train: true`.  
3. Put its id in `sets.studio.sequence` and raise `goal` if you want a longer session.

No app rewrite. The guidance agent picks from that pool.

## Layout

- `app/` — tkinter UI  
- `guidance_agent/` — session coach + catalog  
- `feedback_agent/` — joint math + Nemotron  
- `motion-diffusion-model/` — MDM  
- `3d_pose_estimation/TokenHMR/` — live pose worker  

Do not commit `feedback_agent/api.yaml` or model `.pt` files.

Weights stay on the machine that trained them:

- MDM yoga checkpoint: `motion-diffusion-model/save/surya/model000770007.pt`
- TokenHMR: under `3d_pose_estimation/TokenHMR/datasets/` (not in this repo)
