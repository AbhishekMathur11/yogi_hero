import warnings
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

from motion_gate import MotionGate, MotionGateConfig

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)


class TokenHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        from lib.models import load_tokenhmr

        model, _ = load_tokenhmr(checkpoint_path=cfg.checkpoint, \
                                 model_cfg=cfg.model_config, \
                                 is_train_state=False, is_demo=True)

        self.model = model
        self.model.eval()
        mg_cfg = getattr(cfg, 'motion_gate', None)
        self.motion_gate = MotionGate(mg_cfg) if mg_cfg is not None else MotionGate(cfg)

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        self.motion_gate.apply(model_out['pred_smpl_params'], model_out['pred_cam'], x)

        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class PHALP_Prime_TokenHMR(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = TokenHMRPredictor(self.cfg)

@dataclass
class Human4DConfig(FullConfig):
    motion_gate: MotionGateConfig = field(default_factory=MotionGateConfig)

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    """Main function for running the PHALP tracker."""

    phalp_tracker = PHALP_Prime_TokenHMR(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
