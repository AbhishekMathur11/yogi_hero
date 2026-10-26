#!/usr/bin/env python3
"""Short TokenHMR finetune on a local MOYO npz. Writes a new ckpt next to zeroshot weights."""
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
from torch.utils.data import DataLoader

from lib.datasets.image_dataset import ImageDataset
from lib.models.tokenhmr import TokenHMR
from lib.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ZERO = os.path.join(ROOT, 'datasets/data/checkpoints')
OUT = os.path.join(ROOT, 'datasets/data/checkpoints_finetune')


class _ImgOnly(pl.LightningDataModule):
    def __init__(self, cfg, npz, img_dir, batch_size, num_workers):
        super().__init__()
        self.cfg, self.npz, self.img_dir = cfg, npz, img_dir
        self.bs, self.nw = batch_size, num_workers

    def setup(self, stage=None):
        self.ds = ImageDataset(
            self.cfg, dataset_file=self.npz, img_dir=self.img_dir,
            train=True, dataset_name='MOYO-BHUJANGASANA',
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds, batch_size=self.bs, shuffle=True, drop_last=True,
            num_workers=self.nw, persistent_workers=self.nw > 0,
        )


class _MoyoFit(TokenHMR):
    """Same losses, but the loader is images-only (no mocap / GAN)."""

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        self.log('train/loss', output['losses']['loss'], on_step=True, prog_bar=True)
        return output


def main():
    os.chdir(ROOT)
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(os.path.join(ZERO, 'model_config.yaml'))
    cfg.defrost()
    cfg.ckpt_path = os.path.join(ZERO, 'tokenhmr_model_latest.ckpt')
    cfg.MODEL.FROZEN_LEARNED = True
    cfg.MODEL.LOOSE_SUP = False
    cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0
    cfg.TRAIN.LR = 1e-6
    cfg.TRAIN.BATCH_SIZE = 4
    cfg.TRAIN.WEIGHT_DECAY = 1e-4
    cfg.GENERAL.LOG_STEPS = 10 ** 9
    cfg.MODEL.SMPL_HEAD.TOKENIZER.TRAIN_TOK = False
    cfg.freeze()

    npz = os.path.join(ROOT, 'datasets/data/moyo/moyo_bhujangasana.npz')
    img_dir = os.path.join(ROOT, 'datasets/moyo_toolkit/data')

    model = _MoyoFit(cfg, init_renderer=False, is_train_state=True)
    ckpt = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    log.info(f'loaded zeroshot  missing={len(missing)} unexpected={len(unexpected)}')
    model.frozen_backbone()

    os.makedirs(OUT, exist_ok=True)
    dm = _ImgOnly(cfg, npz, img_dir, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=2)
    cb = pl.callbacks.ModelCheckpoint(
        dirpath=OUT, filename='moyo_bhujangasana', save_last=True,
        every_n_train_steps=200, save_weights_only=True,
    )
    trainer = pl.Trainer(
        accelerator='gpu', devices=1, max_steps=400,
        callbacks=[cb], logger=False, enable_checkpointing=True,
        precision=16,
    )
    trainer.fit(model, datamodule=dm)
    dest = os.path.join(OUT, 'moyo_bhujangasana.ckpt')
    trainer.save_checkpoint(dest)
    print(f'saved {dest}')


if __name__ == '__main__':
    main()
