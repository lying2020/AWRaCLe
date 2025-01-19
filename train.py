import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset, ValDataset
from net.model import AWRaCLe
from utils.schedulers import LinearWarmupCosineAnnealingLR

from options import options as opt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


from pytorch_lightning.strategies.ddp import DDPStrategy


class AWRaCLeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AWRaCLe(decoder=True)
        self.loss_fn = nn.L1Loss()
        for name, param in self.named_parameters():
            if 'clip' in name:
                param.requires_grad = False

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch, degrad_context, clean_context) = batch

        restored = self.net(degrad_patch, [degrad_context, clean_context])


        loss = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch, degrad_context, clean_context) = batch
        degrad_context = degrad_context.float()
        clean_context = clean_context.float()
        restored = self.net(degrad_patch, [degrad_context, clean_context])

        psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        score_psnr = psnr(restored, clean_patch)

        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        score_ssim = ssim(restored, clean_patch)

        loss = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log_dict({"psnr": score_psnr, "ssim": score_ssim})
        if de_id == 3:
            self.log_dict({"valid_loss": loss, "psnr_rain": score_psnr, "ssim_rain": score_ssim})
        elif de_id == 4:
            self.log_dict({"valid_loss": loss, "psnr_haze": score_psnr, "ssim_haze": score_ssim})
        elif de_id == 5:
            self.log_dict({"valid_loss": loss, "psnr_snow": score_psnr, "ssim_snow": score_ssim})

        if de_id == 6:
            self.log_dict({"valid_loss": loss, "psnr_heavyrain": score_psnr, "ssim_heavyrain": score_ssim})
        elif de_id == 7:
            self.log_dict({"valid_loss": loss, "psnr_heavyhaze": score_psnr, "ssim_heavyhaze": score_ssim})
        if de_id == 8:
            self.log_dict({"valid_loss": loss, "psnr_heavysnow": score_psnr, "ssim_heavysnow": score_ssim})
        return loss

    def lr_scheduler_step(self, scheduler, metric, *args, **kwargs):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="AWRaCLe-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = TrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    valset = ValDataset(opt)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False,
                           drop_last=False, num_workers=opt.num_workers)

    model = AWRaCLeModel()

    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, strategy=DDPStrategy(find_unused_parameters=False), logger=logger,
                         callbacks=[checkpoint_callback])
    if opt.resume:
         trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader, ckpt_path=opt.ckpt_path)
    else:
         trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()
