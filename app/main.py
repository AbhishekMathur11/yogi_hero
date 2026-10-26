"""Yoga Teacher desktop coach. Run with the mdm conda env.

Left: MDM reference clip. Right: RealSense. Bottom: math score + Nemotron line.
TokenHMR pose lives in the tkhmr env as a worker process so torch stacks stay apart.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feedback_agent"))
sys.path.insert(0, str(ROOT / "guidance_agent"))

from feedback import compare, nemotron_coach  # noqa: E402
from guidance import Guidance  # noqa: E402
from orbs import ThinkingOrbs  # noqa: E402

ASANAS_PATH = ROOT / "guidance_agent" / "asanas.json"
PLAN_PATH = ROOT / "guidance_agent" / "plan.json"
MDM_ROOT = ROOT / "motion-diffusion-model"
MDM_CKPT = MDM_ROOT / "save/surya/model000770007.pt"
IPC = ROOT / "app" / ".live"
CONDA = Path.home() / "miniconda3" / "envs"
MDM_PY = CONDA / "mdm" / "bin" / "python"
TKHMR_PY = CONDA / "tkhmr" / "bin" / "python"
TOKENHMR = ROOT / "3d_pose_estimation" / "TokenHMR"

BG, PANEL, INK, MUTED, ACCENT = "#0c0b0a", "#161310", "#f3ead8", "#9c8f7c", "#e8c9a0"


def _placeholder(w, h, text):
    img = np.full((h, w, 3), 18, dtype=np.uint8)
    cv2.putText(img, text, (28, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 160, 140), 1, cv2.LINE_AA)
    return img


class Latest:
    def __init__(self):
        self._lock = threading.Lock()
        self.frame = None
        self.label = "idle"
        self.stop = threading.Event()

    def set(self, frame, label):
        with self._lock:
            self.frame = None if frame is None else frame.copy()
            self.label = label

    def get(self):
        with self._lock:
            if self.frame is None:
                return None, self.label
            return self.frame.copy(), self.label


def _open_camera(width, height, fps, webcam):
    if webcam >= 0:
        cap = cv2.VideoCapture(webcam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            raise RuntimeError("webcam failed")

        def read():
            ok, frame = cap.read()
            return frame if ok else None

        return read, cap.release, "webcam"

    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipe.start(cfg)
    align = rs.align(rs.stream.color)

    def read():
        frames = align.process(pipe.wait_for_frames())
        return np.asanyarray(frames.get_color_frame().get_data())

    return read, pipe.stop, "realsense"


def _camera_loop(buf: Latest, width, height, fps, webcam):
    try:
        read, close, kind = _open_camera(width, height, fps, webcam)
    except Exception as exc:
        buf.set(_placeholder(width, height, "no camera"), str(exc)[:42])
        return
    try:
        while not buf.stop.is_set():
            frame = read()
            if frame is None:
                time.sleep(0.02)
                continue
            buf.set(frame, kind)
            try:
                IPC.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(IPC / "live.jpg"), frame)
            except Exception:
                pass
    finally:
        close()


def _video_loop(buf: Latest, path_holder):
    cap = None
    current = None
    while not buf.stop.is_set():
        path = path_holder[0]
        if path is None or not Path(path).is_file():
            caption = path_holder[1] if len(path_holder) > 1 and path_holder[1] else "waiting for guide"
            buf.set(_placeholder(640, 400, caption[:42]), caption)
            time.sleep(0.1)
            continue
        if current != path:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(str(path))
            current = path
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        caption = path_holder[1] if len(path_holder) > 1 and path_holder[1] else "SMPL guide"
        buf.set(frame, caption)
        time.sleep(1.0 / max(cap.get(cv2.CAP_PROP_FPS) or 20.0, 1.0))
    if cap is not None:
        cap.release()


def load_catalog():
    return json.loads(ASANAS_PATH.read_text())


def write_plan(set_id, catalog):
    asanas = catalog["asanas"]
    info = catalog["sets"][set_id]
    pool = [
        name
        for name in info["sequence"]
        if asanas.get(name, {}).get("plausible") and asanas.get(name, {}).get("in_train")
    ]
    if not pool:
        raise RuntimeError("no asana in this set has a usable 3D body")
    goal = min(int(info.get("goal") or len(pool)), len(pool))
    PLAN_PATH.write_text(json.dumps({"set": set_id, "goal": goal, "pool": pool, "sequence": pool}, indent=2))
    return pool


def _prompt_key(prompt):
    first = (prompt or "").split(".")[0].strip()
    return first.replace(" ", "_")[:48]


def _sample_folder(asana, prompt):
    surya = MDM_ROOT / "save" / "surya"
    cache = surya / "app_cache" / asana
    if (cache / "output_mesh" / "body.mp4").is_file():
        return cache
    key = _prompt_key(prompt)
    hits = []
    for folder in surya.glob("samples_*"):
        name = folder.name
        if asana in name or (key and key[:32] in name):
            hits.append(folder)
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def find_clip(asana, prompt):
    folder = _sample_folder(asana, prompt)
    if folder is None:
        return None, None
    body = folder / "output_mesh" / "body.mp4"
    npy = folder / "results.npy"
    if body.is_file():
        return body, npy if npy.is_file() else None
    return None, npy if npy.is_file() else None


def asana_line(entry):
    sanskrit = entry.get("sanskrit") or entry.get("title") or ""
    english = entry.get("english") or ""
    if english and english.lower() not in sanskrit.lower():
        return f"{sanskrit}  ·  {english}"
    return sanskrit


def ensure_body_video(asana, prompt, on_log=None):
    """Return the pyrender body.mp4 only. Never the matplotlib stick clip."""
    folder = _sample_folder(asana, prompt)
    if folder is None:
        return None, None
    body = folder / "output_mesh" / "body.mp4"
    smpl = folder / "output_mesh" / "smpl_params.npy"
    npy = folder / "results.npy"
    if body.is_file():
        return body, npy if npy.is_file() else None
    if not smpl.is_file():
        return None, None
    proc = subprocess.run(
        [str(MDM_PY), str(ROOT / "app" / "mesh_view.py"), "--smpl", str(smpl), "--out", str(body)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not body.is_file():
        return None, None
    return body, npy if npy.is_file() else None


def target_joints(npy_path):
    if npy_path is None or not Path(npy_path).is_file():
        return None
    data = np.load(npy_path, allow_pickle=True)
    motion = data.item()["motion"] if hasattr(data, "item") else data
    if isinstance(motion, dict):
        motion = motion["motion"]
    # (N, 22, 3, T)
    t = motion.shape[-1] // 2
    return motion[0, :, :, t]


def start_pose_worker():
    IPC.mkdir(parents=True, exist_ok=True)
    for name in ("worker.stop", "worker.ready", "live_overlay.jpg", "live_joints.npy"):
        p = IPC / name
        if p.exists():
            p.unlink()
    log = open(IPC / "worker.log", "w")
    return subprocess.Popen(
        [str(TKHMR_PY), "tokenhmr/pose_worker.py", "--ipc", str(IPC)],
        cwd=str(TOKENHMR),
        stdout=log,
        stderr=log,
    )


class Pane(tk.Frame):
    def __init__(self, parent, title, subtitle):
        super().__init__(parent, bg=PANEL)
        head = tk.Frame(self, bg=PANEL)
        head.pack(fill="x", padx=18, pady=(16, 8))
        tk.Label(head, text=title, fg=INK, bg=PANEL, font=("Georgia", 18)).pack(anchor="w")
        self.sub = tk.Label(head, text=subtitle, fg=MUTED, bg=PANEL, font=("Helvetica", 10))
        self.sub.pack(anchor="w")
        self.canvas = tk.Label(self, bg="#090807")
        self.canvas.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._photo = None

    def show(self, bgr, caption):
        self.sub.config(text=caption)
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        box_w = max(self.canvas.winfo_width(), 360)
        box_h = max(self.canvas.winfo_height(), 240)
        scale = min(box_w / w, box_h / h)
        rgb = cv2.cvtColor(cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale)))), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=self._photo)


class App(tk.Tk):
    def __init__(self, webcam):
        super().__init__()
        self.webcam = webcam
        self.catalog = load_catalog()
        self.guide = None
        self.target = None
        self.clip_path = [None, "SMPL guide"]
        self.live = Latest()
        self.mdm = Latest()
        self.worker = None
        self.busy = False
        self._cam_started = False
        self._loading = False
        self.coach = "Choose a sequence to begin."
        self.score = None
        self._last_speak = 0.0
        self._agent_action = None

        self.title("Yoga Teacher")
        self.configure(bg=BG)
        self.geometry("1440x860")
        self.minsize(1100, 680)

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=24, pady=(18, 6))
        names = tk.Frame(top, bg=BG)
        names.pack(side="left")
        tk.Label(names, text="Yoga Teacher", fg=INK, bg=BG, font=("Georgia", 28)).pack(anchor="w")
        tk.Label(names, text="Hold each asana to move on", fg=MUTED, bg=BG, font=("Georgia", 13)).pack(anchor="w")
        self.status = tk.Label(top, text="ready", fg=MUTED, bg=BG, font=("Helvetica", 11))
        self.status.pack(side="right")

        split = tk.Frame(self, bg=BG)
        split.pack(fill="both", expand=True, padx=20, pady=8)
        split.columnconfigure(0, weight=1)
        split.columnconfigure(1, weight=1)
        split.rowconfigure(0, weight=1)
        self.left = Pane(split, "Guide", "3D body")
        self.right = Pane(split, "You", "You")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        foot = tk.Frame(self, bg=PANEL)
        foot.pack(fill="x", padx=20, pady=(4, 16))
        row = tk.Frame(foot, bg=PANEL)
        row.pack(fill="x", padx=18, pady=(14, 6))
        self.asana_lbl = tk.Label(row, text="—", fg=INK, bg=PANEL, font=("Georgia", 20))
        self.asana_lbl.pack(side="left")
        self.score_lbl = tk.Label(row, text="", fg=ACCENT, bg=PANEL, font=("Helvetica", 16))
        self.score_lbl.pack(side="right")
        self.bar = tk.Canvas(foot, height=6, bg="#2a241e", highlightthickness=0)
        self.bar.pack(fill="x", padx=18, pady=(0, 10))
        self.coach_lbl = tk.Label(
            foot, text=self.coach, fg=INK, bg=PANEL, font=("Georgia", 13), wraplength=900, justify="left"
        )
        self.coach_lbl.pack(anchor="w", padx=18, pady=(0, 12))

        self.overlay = tk.Frame(self, bg=BG)
        self.orbs = ThinkingOrbs(self.overlay, size=148)
        self.orbs.pack(pady=(220, 12))
        self.busy_lbl = tk.Label(self.overlay, text="", fg=ACCENT, bg=BG, font=("Georgia", 16))
        self.busy_lbl.pack()

        self.start_card = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground="#3a3228")
        tk.Label(self.start_card, text="Yoga Teacher", fg=INK, bg=PANEL, font=("Georgia", 26)).pack(pady=(36, 8))
        tk.Button(
            self.start_card,
            text="Start practice",
            font=("Georgia", 15),
            bg=ACCENT,
            fg="#1a140f",
            activebackground=INK,
            activeforeground="#1a140f",
            relief="flat",
            cursor="hand2",
            padx=28,
            pady=12,
            command=lambda: self._begin("studio"),
        ).pack(pady=(8, 36))

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(40, self.tick)
        self.after(80, self._show_start)

        threading.Thread(target=_video_loop, args=(self.mdm, self.clip_path), daemon=True).start()

    def _show_start(self):
        self.start_card.place(relx=0.5, rely=0.46, anchor="center", width=420, height=200)

    def _log(self, msg):
        self.after(0, lambda m=msg: self.status.config(text=m))

    def _start_live(self):
        if self._cam_started:
            return
        self._cam_started = True
        threading.Thread(
            target=_camera_loop, args=(self.live, 640, 480, 30, self.webcam), daemon=True
        ).start()
        if TKHMR_PY.is_file():
            try:
                self.worker = start_pose_worker()
            except Exception as exc:
                self.coach = f"Pose worker failed: {exc}"[:240]

    def _begin(self, set_id):
        self.start_card.place_forget()
        seq = write_plan(set_id, self.catalog)
        self.guide = Guidance(PLAN_PATH)
        self._set_busy("")
        threading.Thread(target=self._boot, args=(seq,), daemon=True).start()

    def _boot(self, seq):
        try:
            self._start_live()
            self._apply_asana(seq[0])
            for _ in range(80):
                frame, _ = self.live.get()
                if frame is not None:
                    break
                time.sleep(0.1)
        except Exception as exc:
            self.coach = str(exc)[:240]
        finally:
            self.after(0, self._session_ready)

    def _session_ready(self):
        if self.guide is not None:
            self.guide.begin_imitate()
            entry = self.catalog["asanas"].get(self.guide.asana or "")
            if entry:
                self.asana_lbl.config(text=asana_line(entry))
        self._clear_busy()

    def _set_busy(self, text=""):
        self.busy = True
        self.busy_lbl.config(text=text)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.orbs.start()

    def _clear_busy(self):
        self.busy = False
        self._loading = False
        self.orbs.stop()
        self.overlay.place_forget()

    def _apply_asana(self, asana, on_log=None):
        entry = self.catalog["asanas"][asana]
        self.clip_path[0] = None
        self.clip_path[1] = "3D body"
        if not entry.get("plausible") or not entry.get("in_train"):
            raise RuntimeError("no usable 3D body for this asana")
        body, npy = ensure_body_video(asana, entry["mdm_prompt"], on_log=on_log)
        if body is None:
            raise RuntimeError("3D body missing")
        self.clip_path[0] = str(body)
        self.target = target_joints(npy)
        self.coach = entry.get("cue") or f"Match {entry['sanskrit']}."
        line = asana_line(entry)
        self.after(0, lambda t=line: self.asana_lbl.config(text=t))

    def _load_asana_clip(self, asana):
        if self._loading:
            return
        self._loading = True
        self.after(0, lambda: self._set_busy(""))
        def work():
            try:
                self._apply_asana(asana)
            except Exception as exc:
                self.coach = str(exc)[:200]
            finally:
                self.after(0, self._session_ready)
        threading.Thread(target=work, daemon=True).start()

    def _fresh_overlay(self):
        overlay = IPC / "live_overlay.jpg"
        if not overlay.is_file():
            return None
        age = time.time() - overlay.stat().st_mtime
        if age > 1.5:
            return None
        return cv2.imread(str(overlay))

    def tick(self):
        mdm, mtag = self.mdm.get()
        live, ltag = self.live.get()
        ov = self._fresh_overlay()
        if ov is not None:
            live, ltag = ov, "TokenHMR overlay"
        self.left.show(mdm if mdm is not None else _placeholder(640, 400, mtag if mtag != "idle" else "no MDM clip"), mtag)
        self.right.show(live if live is not None else _placeholder(640, 400, ltag or "you"), ltag)

        joints_path = IPC / "live_joints.npy"
        if self.guide and self.target is not None and joints_path.is_file():
            try:
                live_j = np.load(joints_path)
                report = compare(live_j, self.target, speak=False)
                self.score = report["score"]
                suggestion = self._agent_action
                self._agent_action = None
                st = self.guide.tick(self.score, suggestion=suggestion)
                entry = self.catalog["asanas"].get(st["asana"] or "")
                title = asana_line(entry) if entry else (st["asana"] or "Practice complete")
                self.asana_lbl.config(text=title)
                report["asana"] = entry.get("sanskrit") if entry else st["asana"]
                report["cue"] = entry.get("cue") if entry else ""
                report["pool"] = list(self.guide.pool)
                report["remaining"] = self.guide.remaining()
                self.score_lbl.config(text=f"{int(self.score * 100)}%")
                w = max(self.bar.winfo_width(), 10)
                self.bar.delete("all")
                self.bar.create_rectangle(0, 0, w * min(self.score, 1.0), 6, fill=ACCENT, outline="")
                now = time.monotonic()
                if now - self._last_speak > 2.5:
                    self._last_speak = now
                    threading.Thread(target=self._speak, args=(report,), daemon=True).start()
                if st.get("action") == "retry":
                    pass
                elif st.get("advanced") and st.get("asana"):
                    self._load_asana_clip(st["asana"])
                elif st.get("phase") == "done":
                    self.coach = "Practice complete."
            except Exception as exc:
                self.status.config(text=str(exc)[:80])

        self.coach_lbl.config(text=self.coach)
        self.after(40, self.tick)

    def _speak(self, report):
        self.status.config(text="Nemotron…")
        try:
            line = nemotron_coach(report)
            self.coach = line.get("text") or self.coach
            act = line.get("action")
            if act in ("retry", "skip", "easier", "advance"):
                self._agent_action = act
        finally:
            self.after(0, lambda: self.status.config(text="live"))

    def on_close(self):
        self.live.stop.set()
        self.mdm.stop.set()
        IPC.mkdir(parents=True, exist_ok=True)
        (IPC / "worker.stop").write_text("1")
        if self.worker is not None:
            try:
                self.worker.terminate()
            except Exception:
                pass
        self.destroy()


def main():
    webcam = -1
    if "--webcam" in sys.argv:
        webcam = int(sys.argv[sys.argv.index("--webcam") + 1])
    App(webcam).mainloop()


if __name__ == "__main__":
    main()
