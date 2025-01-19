import os
import argparse
import subprocess
import time
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.dataset_utils import TestDataset_IC, TestDataset_Folder
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import pytorch_lightning as pl
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from net.model import AWRaCLe


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AWRaCLe(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x, ctx=None):
        if ctx is not None:
            return self.net(x, ctx)
        else:
            return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]


def test_ds(net, dataset, args):
    output_path = testopt.output_path
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    lpips_scores = AverageMeter()
    fid_value = 0

    calc_lpips = args.lpips
    calc_fid = args.fid

    if calc_lpips:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

    if calc_fid:
        fid_metric = FrechetInceptionDistance(feature=2048).cuda()

    stop_count = 501  # Some test sets are huge
    count = 0
    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch, degrad_context, clean_context) in tqdm(testloader):
            if count > stop_count:
                break
            count += 1

            clean_name = clean_name[0].split('/')[-1]

            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            degrad_context, clean_context = degrad_context.cuda(), clean_context.cuda()

            restored = net(degrad_patch, [degrad_context, clean_context])

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)


            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)


            restored = restored.clamp(0, 1)
            clean_patch = clean_patch.clamp(0, 1)


            # Prepare images for LPIPS (range [-1,1])
            restored_lpips = 2 * restored - 1
            clean_patch_lpips = 2 * clean_patch - 1

            # Prepare images for FID/KID (range [0,255], uint8)
            restored_fid = restored
            clean_patch_fid = clean_patch
            restored_fid_uint8 = restored_fid.mul(255).byte()
            clean_patch_fid_uint8 = clean_patch_fid.mul(255).byte()

            if calc_lpips:
                lpips_value = lpips_metric(restored_lpips, clean_patch_lpips)
                lpips_scores.update(lpips_value.item(), N)

            if calc_fid:
                fid_metric.update(restored_fid_uint8, real=False)
                fid_metric.update(clean_patch_fid_uint8, real=True)

            save_image_tensor(restored, output_path + clean_name.replace(clean_name.split('.')[-1], 'png'))

        if calc_fid:
            fid_value = fid_metric.compute().item()

        print("psnr: %.2f, ssim: %.4f, lpips: %.4f, fid: %.4f" % (psnr.avg, ssim.avg, lpips_scores.avg, fid_value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0, help='0: load from json (recommended), '
                                                            '1: load images from folder, expects test_dir with GT/, degraded/ folders')

    parser.add_argument('--test_dir', type=str, default="test/dehaze/", help='root directory where .json files or images are present')
    parser.add_argument('--test_json', type=str, default="test/dehaze/", help='.json file containing test image list')
    parser.add_argument('--output_path', type=str, default="output/", help='restored images save path')
    parser.add_argument('--ckpt_name', type=str, default=".ckpt", help='checkpoint name')
    parser.add_argument('--in_context_dir', type=str, default=None, help='directory of context pairs')
    parser.add_argument('--in_context_file', type=str, default=None, help='.json file to sample in-context pairs')
    parser.add_argument('--lpips', action='store_true', help='compute LPIPS')
    parser.add_argument('--fid', action='store_true', help='compute FID')
    parser.add_argument('--degrad_context', type=str, default=None, help='degraded context image (if you want to pass your own context)')
    parser.add_argument('--clean_context', type=str, default=None, help='clean context image (if you want to pass your own context)')

    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_name

    if testopt.degrad_context is not None and testopt.clean_context is not None:
        pair = (testopt.degrad_context, testopt.clean_context)
    else:
        pair = None

    if testopt.mode == 1:
        dset  = TestDataset_Folder(testopt, pair=pair)
    else:
        dset = TestDataset_IC(testopt, pair=pair)

    print("CKPT name : {}".format(ckpt_path))
    net = Model().load_from_checkpoint(ckpt_path).cuda()
    net.eval()
    print("Loaded!")

    print('Start testing...')
    test_ds(net, dset, testopt)
