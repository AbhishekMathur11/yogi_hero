"""Thinking-orbs 'working' state, drawn in Tk. Port of extras/thinking-orbs orbits."""

from __future__ import annotations

import math
import time
import tkinter as tk

BG = "#0c0b0a"
DOT = "#f4eee4"


def _hash(i, salt):
    x = math.sin(i * 12.9898 + salt * 78.233) * 43758.5453
    return x - math.floor(x)


class ThinkingOrbs(tk.Canvas):
    def __init__(self, parent, size=140, **kwargs):
        super().__init__(parent, width=size, height=size, bg=BG, highlightthickness=0, **kwargs)
        self.size = size
        self._t0 = time.monotonic()
        self._job = None
        self._active = False

    def start(self):
        if self._active:
            return
        self._active = True
        self._tick()

    def stop(self):
        self._active = False
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        self.delete("all")

    def _tick(self):
        if not self._active:
            return
        t = (time.monotonic() - self._t0) * 1.885
        self.delete("all")
        s = self.size
        cx = cy = s / 2
        R = (s / 2) * 0.82
        orbit_n, ghost_n, particles = 8, 28, 3
        yaw = t * 0.12
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        for orb in range(orbit_n):
            h1, h2, h3 = _hash(orb, 1.7), _hash(orb, 5.2), _hash(orb, 8.9)
            ro = R * (0.45 + 0.52 * h1)
            th = h1 * 2 * math.pi
            phi = math.acos(2 * h2 - 1)
            nx = math.sin(phi) * math.cos(th)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(th)
            ux, uy = -ny, nx
            ul = max(1e-6, math.sqrt(ux * ux + uy * uy))
            ux, uy = ux / ul, uy / ul
            vx, vy, vz = ny * 0 - nz * uy, nz * ux - nx * 0, nx * uy - ny * ux
            speed = (0.25 + 0.55 * h3) * (1 if h3 > 0.5 else -1)
            for k in range(ghost_n):
                a = (k / ghost_n) * 2 * math.pi
                x = (ux * math.cos(a) + vx * math.sin(a)) * ro
                y = (uy * math.cos(a) + vy * math.sin(a)) * ro
                z = (0 * math.cos(a) + vz * math.sin(a)) * ro
                px = cx + x * cos_y - z * sin_y
                py = cy + y
                depth = (z / ro + 1) / 2
                r = 0.9
                a_alpha = int(255 * 0.5 * (0.35 + 0.65 * depth))
                color = f"#{a_alpha:02x}{a_alpha:02x}{int(a_alpha * 0.92):02x}"
                self.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="")
            for p in range(particles):
                a = t * speed + p * 2 * math.pi / particles
                x = (ux * math.cos(a) + vx * math.sin(a)) * ro
                y = (uy * math.cos(a) + vy * math.sin(a)) * ro
                z = (0 * math.cos(a) + vz * math.sin(a)) * ro
                px = cx + x * cos_y - z * sin_y
                py = cy + y
                rr = 2.2
                self.create_oval(px - rr, py - rr, px + rr, py + rr, fill=DOT, outline="")
        self._job = self.after(33, self._tick)
