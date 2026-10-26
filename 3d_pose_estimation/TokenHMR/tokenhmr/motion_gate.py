"""Hold SMPL + camera until the person actually changes pose.

Copied out of track.py so the live RGB-D loop can use the same gate without PHALP.
Motion is measured against the *locked* crop, not the previous frame.
"""

from dataclasses import dataclass

import torch

try:
    from phalp.utils import get_pylogger
    log = get_pylogger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)

_IMAGENET_MEAN = (123.675, 116.280, 103.530)
_IMAGENET_STD = (58.395, 57.120, 57.375)


def _mask_centroid(mask: torch.Tensor):
    weight = mask.clamp(0, 1)
    mass = weight.sum().clamp(min=1.0)
    h, w = mask.shape
    ys = torch.arange(h, device=mask.device, dtype=mask.dtype)
    xs = torch.arange(w, device=mask.device, dtype=mask.dtype)
    cy = (weight.sum(dim=1) * ys).sum() / mass
    cx = (weight.sum(dim=0) * xs).sum() / mass
    return cy, cx


def _roll_hw(t: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    return torch.roll(t, shifts=(int(round(float(dy))), int(round(float(dx)))), dims=(-2, -1))


def _unnormalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    mean = rgb.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (rgb * std + mean).clamp(0, 255) / 255.0


def _mean_geodesic_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    R_rel = torch.matmul(R_a.transpose(-1, -2), R_b)
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(cos)).mean(dim=-1)


def _rotmats(pred_smpl: dict, key: str) -> torch.Tensor:
    x = pred_smpl[key]
    if x.ndim == 4:
        return x
    return x.reshape(x.shape[0], -1, 3, 3)


@dataclass
class MotionGateConfig:
    enable: bool = True
    pixel_thresh: float = 0.05
    pose_thresh_deg: float = 12.0
    unlock_frames: int = 2
    lock_cam: bool = True
    log_every: int = 30


class MotionGate:
    """Hold the last accepted SMPL + camera until the person actually changes pose."""

    def __init__(self, cfg) -> None:
        self.enable = bool(getattr(cfg, "enable", True))
        self.pixel_thresh = float(getattr(cfg, "pixel_thresh", 0.05))
        self.pose_thresh_deg = float(getattr(cfg, "pose_thresh_deg", 12.0))
        self.unlock_frames = int(getattr(cfg, "unlock_frames", 2))
        self.lock_cam = bool(getattr(cfg, "lock_cam", True))
        self.log_every = int(getattr(cfg, "log_every", 30))
        self.last_img_motion = None
        self.last_pose_motion = None
        self.reset()

    def reset(self) -> None:
        self._locked_rgb = None
        self._locked_mask = None
        self._locked_smpl = None
        self._locked_cam = None
        self._unlock_count = None
        self._frame = 0
        self._n_locked = 0
        self._n_unlocked = 0

    def _store_lock(self, pred_smpl: dict, pred_cam: torch.Tensor,
                    rgb: torch.Tensor, mask: torch.Tensor) -> None:
        self._locked_smpl = {k: v.detach().clone() for k, v in pred_smpl.items()}
        self._locked_cam = pred_cam.detach().clone()
        self._locked_rgb = rgb.detach().clone()
        self._locked_mask = mask.detach().clone()
        self._unlock_count = torch.zeros(rgb.shape[0], dtype=torch.long, device=pred_cam.device)

    def _image_motion(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = rgb.shape[0]
        scores = rgb.new_zeros(b)
        if self._locked_rgb is None or self._locked_rgb.shape[0] != b:
            return scores + 1.0
        for i in range(b):
            curr_m = mask[i]
            lock_m = self._locked_mask[i]
            cy_c, cx_c = _mask_centroid(curr_m)
            cy_l, cx_l = _mask_centroid(lock_m)
            lock_rgb_a = _roll_hw(self._locked_rgb[i], cy_c - cy_l, cx_c - cx_l)
            lock_m_a = _roll_hw(lock_m, cy_c - cy_l, cx_c - cx_l)
            valid = (curr_m > 0.5) & (lock_m_a > 0.5)
            if int(valid.sum()) < 32:
                scores[i] = 1.0
                continue
            scores[i] = (rgb[i] - lock_rgb_a).abs().mean(dim=0)[valid].mean()
        return scores

    def _pose_motion(self, pred_smpl: dict) -> torch.Tensor:
        body = _mean_geodesic_deg(_rotmats(self._locked_smpl, "body_pose"),
                                 _rotmats(pred_smpl, "body_pose"))
        if "global_orient" not in pred_smpl or "global_orient" not in self._locked_smpl:
            return body
        orient = _mean_geodesic_deg(_rotmats(self._locked_smpl, "global_orient"),
                                   _rotmats(pred_smpl, "global_orient"))
        return 0.75 * body + 0.25 * orient

    def apply(self, pred_smpl: dict, pred_cam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Mutate pred_smpl / pred_cam in place. Returns True per batch item if the lock was refreshed."""
        b = pred_cam.shape[0]
        device = pred_cam.device
        if not self.enable:
            return torch.ones(b, dtype=torch.bool, device=device)

        rgb = _unnormalize_rgb(x[:, :3])
        mask = (x[:, 3] > 0.5).to(rgb.dtype) if x.shape[1] >= 4 else torch.ones(
            b, rgb.shape[-2], rgb.shape[-1], device=device, dtype=rgb.dtype
        )
        if self._locked_smpl is None or self._locked_cam.shape[0] != b:
            self._store_lock(pred_smpl, pred_cam, rgb, mask)
            self._frame += 1
            self.last_img_motion = rgb.new_ones(b)
            self.last_pose_motion = rgb.new_zeros(b)
            return torch.ones(b, dtype=torch.bool, device=device)

        img_motion = self._image_motion(rgb, mask)
        pose_motion = self._pose_motion(pred_smpl)
        self.last_img_motion = img_motion.detach()
        self.last_pose_motion = pose_motion.detach()
        moving = (img_motion > self.pixel_thresh) | (pose_motion > self.pose_thresh_deg)
        self._unlock_count = torch.where(
            moving,
            self._unlock_count + 1,
            torch.zeros_like(self._unlock_count),
        )
        accept = self._unlock_count >= self.unlock_frames

        for i in range(b):
            if bool(accept[i]):
                for k, v in pred_smpl.items():
                    self._locked_smpl[k][i] = v[i].detach()
                self._locked_cam[i] = pred_cam[i].detach()
                self._locked_rgb[i] = rgb[i].detach()
                self._locked_mask[i] = mask[i].detach()
                self._unlock_count[i] = 0
                self._n_unlocked += 1
            else:
                for k in pred_smpl:
                    pred_smpl[k][i] = self._locked_smpl[k][i]
                if self.lock_cam:
                    pred_cam[i] = self._locked_cam[i]
                self._n_locked += 1

        self._frame += 1
        if self.log_every > 0 and self._frame % self.log_every == 0:
            locked_now = int((~accept).sum().item())
            log.info(
                f"motion_gate frame={self._frame} "
                f"img={['%.4f' % v for v in img_motion.detach().cpu().tolist()]} "
                f"pose_deg={['%.2f' % v for v in pose_motion.detach().cpu().tolist()]} "
                f"img_thr={self.pixel_thresh} pose_thr={self.pose_thresh_deg} "
                f"locked={locked_now}/{b} held={self._n_locked} opened={self._n_unlocked}"
            )
        return accept
