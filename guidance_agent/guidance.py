"""Session coach: pick the next asana from a catalog pool, never invent one.

plan.json is the allowed set. Nemotron may suggest retry / skip / easier / advance.
Python checks the id is in the pool and has not already been completed.
"""

import json
import time
from pathlib import Path

HOLD_SCORE = 0.85
HOLD_SECONDS = 1.5
STUCK_SECONDS = 12.0
PLAN_PATH = Path(__file__).with_name("plan.json")
CATALOG_PATH = Path(__file__).with_name("asanas.json")
ACTIONS = ("retry", "skip", "easier", "advance")


class Guidance:
    def __init__(self, plan_path=PLAN_PATH, catalog_path=CATALOG_PATH,
                 hold_score=HOLD_SCORE, hold_seconds=HOLD_SECONDS, stuck_seconds=STUCK_SECONDS):
        plan = json.loads(Path(plan_path).read_text())
        catalog = json.loads(Path(catalog_path).read_text())
        self.catalog = catalog.get("asanas", {})
        pool = list(plan.get("pool") or plan.get("sequence") or [])
        self.pool = [a for a in pool if a in self.catalog]
        if not self.pool:
            raise ValueError("plan pool is empty")
        self.goal = int(plan.get("goal") or len(self.pool))
        self.hold_score = hold_score
        self.hold_seconds = hold_seconds
        self.stuck_seconds = stuck_seconds
        self.completed = []
        self.current = self.pool[0]
        self.phase = "demo"
        self._hold_since = None
        self._stuck_since = None
        self._retries = 0

    @property
    def asana(self):
        if self.phase == "done":
            return None
        return self.current

    def begin_imitate(self):
        if self.phase == "demo":
            self.phase = "imitate"
            self._hold_since = None
            self._stuck_since = None
        return self.state()

    def remaining(self):
        return [a for a in self.pool if a not in self.completed and a != self.current]

    def tick(self, score, suggestion=None, now=None):
        if self.phase == "done":
            return self.state(score=score)
        if self.phase == "demo":
            self.phase = "imitate"
        now = time.monotonic() if now is None else now
        suggestion = suggestion if suggestion in ACTIONS else None

        if score >= self.hold_score:
            self._stuck_since = None
            if self._hold_since is None:
                self._hold_since = now
            held = now - self._hold_since
            if held >= self.hold_seconds or suggestion == "advance":
                return self._finish_current(score, why="held")
            return self.state(score=score, held_seconds=round(held, 2))

        self._hold_since = None
        if self._stuck_since is None:
            self._stuck_since = now
        stuck = now - self._stuck_since
        stuck_long = stuck >= self.stuck_seconds

        if suggestion == "retry" or (stuck_long and self._retries < 1):
            return self._retry(score)
        if suggestion in ("skip", "easier") or (stuck_long and self._retries >= 1):
            nxt = self._easier() if suggestion == "easier" else None
            return self._skip(score, prefer=nxt)
        return self.state(score=score, held_seconds=0.0, stuck_seconds=round(stuck, 1))

    def _retry(self, score):
        self._retries += 1
        self._hold_since = None
        self._stuck_since = None
        self.phase = "imitate"
        return self.state(score=score, advanced=False, action="retry")

    def _easier(self):
        cur = self.catalog.get(self.current, {})
        d0 = int(cur.get("difficulty") or 2)
        fam = cur.get("family")
        cands = []
        for name in self.remaining():
            ent = self.catalog.get(name, {})
            d = int(ent.get("difficulty") or 2)
            if d < d0:
                cands.append((0 if ent.get("family") == fam else 1, d, name))
        if not cands:
            return None
        cands.sort()
        return cands[0][2]

    def _skip(self, score, prefer=None):
        self.completed.append(self.current)
        return self._goto(score, prefer or (self.remaining()[0] if self.remaining() else None), why="skip")

    def _finish_current(self, score, why="held"):
        self.completed.append(self.current)
        return self._goto(score, self.remaining()[0] if self.remaining() else None, why=why)

    def _goto(self, score, nxt, why):
        self._hold_since = None
        self._stuck_since = None
        self._retries = 0
        if nxt is None or len(self.completed) >= self.goal:
            self.phase = "done"
            self.current = None
            return self.state(score=score, advanced=True, action="done")
        self.current = nxt
        self.phase = "demo"
        return self.state(score=score, advanced=True, action=why)

    def state(self, score=None, held_seconds=0.0, stuck_seconds=0.0, advanced=False, action=None):
        return {
            "phase": self.phase,
            "asana": self.asana,
            "next": (self.remaining()[0] if self.remaining() else None),
            "completed": list(self.completed),
            "score": score,
            "held_seconds": held_seconds,
            "stuck_seconds": stuck_seconds,
            "advanced": advanced,
            "action": action,
            "speech": nemotron_line(self.phase, self.asana),
        }


def nemotron_line(phase, asana):
    if phase == "done":
        text = "Practice complete."
    elif phase == "demo":
        text = f"This is {asana}."
    else:
        text = f"Match {asana}."
    return {"provider": "nemotron_placeholder", "text": text}


if __name__ == "__main__":
    guide = Guidance()
    shown = guide.state()
    assert shown["asana"] == "cobra" and shown["phase"] == "demo"
    guide.begin_imitate()
    early = guide.tick(0.9, now=0.0)
    assert early["advanced"] is False
    moved = guide.tick(0.9, now=0.0 + HOLD_SECONDS)
    assert moved["phase"] == "done" and moved["asana"] is None
    print(moved["action"], moved["speech"]["text"])
