"""Compare a live 22-joint pose to a reference pose and return one correction.

Joints are (22, 3), Y up, in HumanML3D / SMPL order:
pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2,
left_ankle, right_ankle, spine3, left_foot, right_foot, neck,
left_collar, right_collar, head, left_shoulder, right_shoulder,
left_elbow, right_elbow, left_wrist, right_wrist.

nemotron_coach calls Nebius Token Factory. local_coach is the fallback line.
"""

from pathlib import Path
import json
import numpy as np

_API_FILE = Path(__file__).with_name("api.yaml")
_NEBIUS_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
_NEBIUS_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Name used by the scorer -> (point, vertex, point) in the 22-joint list.
# Spine is separate: degrees the chest leans away from upright.
ANGLE_JOINTS = {
    "left_knee": (1, 4, 7),
    "right_knee": (2, 5, 8),
    "left_hip": (3, 1, 4),
    "right_hip": (3, 2, 5),
    "left_elbow": (16, 18, 20),
    "right_elbow": (17, 19, 21),
    "left_shoulder": (12, 16, 18),
    "right_shoulder": (12, 17, 19),
    "left_ankle": (4, 7, 10),
    "right_ankle": (5, 8, 11),
}

HOLD_SCORE = 0.85


def _angle_deg(a, b, c):
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _spine_lean_deg(joints):
    chest = np.asarray(joints[9], dtype=float) - np.asarray(joints[0], dtype=float)
    norm = np.linalg.norm(chest)
    if norm < 1e-8:
        return 0.0
    up = chest / norm
    cos = np.clip(up[1], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def joint_angles(joints):
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (22, 3):
        raise ValueError(f"expected joints shape (22, 3), got {joints.shape}")
    angles = {name: _angle_deg(joints[i], joints[j], joints[k]) for name, (i, j, k) in ANGLE_JOINTS.items()}
    angles["spine"] = _spine_lean_deg(joints)
    return angles


def compare(live, target, tolerance_deg=25.0, speak=False):
    """Score in [0, 1]. 1 means every measured angle matches the reference."""
    live_angles = joint_angles(live)
    target_angles = joint_angles(target)
    rows = {}
    parts = []
    worst_name = None
    worst_err = -1.0
    for name in live_angles:
        err = abs(live_angles[name] - target_angles[name])
        rows[name] = {
            "live_deg": round(live_angles[name], 1),
            "target_deg": round(target_angles[name], 1),
            "error_deg": round(err, 1),
        }
        parts.append(max(0.0, 1.0 - err / tolerance_deg))
        if err > worst_err:
            worst_name = name
            worst_err = err
    score = float(np.mean(parts))
    report = {
        "score": round(score, 3),
        "hold_ready": score >= HOLD_SCORE,
        "worst": rows[worst_name] | {"joint": worst_name},
        "angles": rows,
    }
    report["coach"] = local_coach(report) if not speak else nemotron_coach(report)
    return report


def local_coach(report):
    worst = report.get("worst") or {}
    name = worst.get("joint")
    if not name or worst.get("error_deg", 0) < 8:
        text = "Hold this pose."
    elif name == "spine":
        text = "Lift your chest." if worst["live_deg"] > worst["target_deg"] else "Fold forward a little more."
    else:
        spoken = name.replace("_", " ")
        text = (
            f"Straighten your {spoken} a little."
            if worst["live_deg"] < worst["target_deg"]
            else f"Bend your {spoken} a little."
        )
    return {"provider": "local", "text": text}


def _api_key():
    if not _API_FILE.exists():
        return None
    raw = _API_FILE.read_text().strip()
    if not raw:
        return None
    if raw.startswith("{") or ":" in raw.split("\n", 1)[0]:
        try:
            data = json.loads(raw)
            return data.get("api_key") or data.get("key")
        except json.JSONDecodeError:
            for line in raw.splitlines():
                if line.lower().startswith("api_key"):
                    return line.split(":", 1)[-1].strip().strip('"')
            return raw
    return raw


def _parse_coach_json(raw):
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            say = str(data.get("say") or data.get("text") or "").strip()
            action = data.get("action")
            if action not in ("retry", "skip", "easier", "advance", None):
                action = None
            if say:
                return {"provider": "nemotron", "text": say.split("\n")[0][:180], "action": action}
        except json.JSONDecodeError:
            pass
    line = raw.split("\n")[0].strip()
    if line:
        return {"provider": "nemotron", "text": line[:180], "action": None}
    return None


def nemotron_coach(report):
    """Spoken line plus an optional action. Pose ids stay in the catalog."""
    local = local_coach(report)
    local["action"] = None
    key = _api_key()
    if not key:
        return local
    try:
        import urllib.request

        body = json.dumps({
            "model": _NEBIUS_MODEL,
            "temperature": 0.2,
            "max_tokens": 80,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a calm yoga teacher. Reply with JSON only: "
                        '{"say":"one short correction","action":"retry|skip|easier|advance"|null}. '
                        "Use the Sanskrit name. Do not invent an asana. "
                        "action=retry same pose, skip leave it, easier a simpler pose in the pool, "
                        "advance only if they already match. No lists."
                    ),
                },
                {
                    "role": "user",
                    "content": "Target pose and joint errors:\n" + json.dumps({
                        "asana": report.get("asana"),
                        "cue": report.get("cue"),
                        "score": report.get("score"),
                        "worst": report.get("worst"),
                        "pool": report.get("pool"),
                        "remaining": report.get("remaining"),
                    }),
                },
            ],
        }).encode()
        req = urllib.request.Request(
            _NEBIUS_URL,
            data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode())
        parsed = _parse_coach_json(payload["choices"][0]["message"]["content"])
        if parsed:
            return parsed
    except Exception:
        pass
    return local


def standing_reference():
    """Simple upright 22-joint stand-in when MDM has no clip."""
    return _standing_stick()


def _standing_stick():
    j = np.zeros((22, 3))
    j[0] = [0, 1.0, 0]
    j[1] = [-0.12, 0.95, 0]
    j[2] = [0.12, 0.95, 0]
    j[3] = [0, 1.15, 0]
    j[4] = [-0.12, 0.5, 0]
    j[5] = [0.12, 0.5, 0]
    j[6] = [0, 1.3, 0]
    j[7] = [-0.12, 0.08, 0]
    j[8] = [0.12, 0.08, 0]
    j[9] = [0, 1.45, 0]
    j[10] = [-0.12, 0.0, 0.08]
    j[11] = [0.12, 0.0, 0.08]
    j[12] = [0, 1.55, 0]
    j[15] = [0, 1.72, 0]
    j[16] = [-0.22, 1.45, 0]
    j[17] = [0.22, 1.45, 0]
    j[18] = [-0.48, 1.45, 0]
    j[19] = [0.48, 1.45, 0]
    j[20] = [-0.72, 1.45, 0]
    j[21] = [0.72, 1.45, 0]
    return j


if __name__ == "__main__":
    target = _standing_stick()
    same = compare(target, target)
    bent = target.copy()
    bent[4] = [-0.12, 0.55, 0.25]
    off = compare(bent, target)
    assert same["score"] == 1.0, same
    assert off["score"] < same["score"], off
    assert off["worst"]["joint"] == "left_knee", off["worst"]
    print(off["score"], off["coach"]["provider"], off["coach"]["text"])
