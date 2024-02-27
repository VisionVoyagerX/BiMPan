from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from data_loader.DataLoader import DIV2K, GaoFen2, Sev2Mod, WV3, GaoFen2panformer
from BiMPan import BiMPan
from utils import *


def main():
    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize DataLoader
    train_dataset = WV3(
        Path("/home/nick/VScode-workspace/Pansharpening_datasets/WV3_Dataset/train/train_wv3-001.h5"), transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])  # /home/ubuntu/project
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=4, shuffle=True, drop_last=True)

    validation_dataset = WV3(
        Path("/home/nick/VScode-workspace/Pansharpening_datasets/WV3_Dataset/val/valid_wv3.h5"))
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=1, shuffle=True)

    test_dataset = WV3(
        Path("/home/nick/VScode-workspace/Pansharpening_datasets/WV3_Dataset/test/test_wv3_multiExm1.h5"))
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics

    # Initialize Model, optimizer, criterion and metrics
    model = BiMPan(ms_channels=8, mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                   pan_std=train_dataset.pan_std.to(device)).to(device)

    optimizer = Adam(model.parameters(), lr=0.0003,
                     betas=(0.9, 0.999))

    criterion = MSELoss().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    tr_report_loss = 0
    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []
    best_eval_psnr = 0
    best_test_psnr = 0
    current_daytime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    steps = 250000
    save_interval = 1000
    report_interval = 50
    test_intervals = [40000, 60000, 100000,
                      140000, 160000, 200000]
    evaluation_interval = [40000, 60000, 100000,
                           140000, 160000, 200000]

    val_steps = 100

    # Model summary
    pan_example = torch.randn(
        (1, 1, 256, 256)).to(device)
    mslr_example = torch.randn(
        (1, 8, 64 * 4, 64 * 4)).to(device)

    summary(model, pan_example, mslr_example, verbose=1)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    lr_decay_intervals = 50000

    continue_from_checkpoint = True
    checkpoint_path = '/home/nick/VScode-workspace/BiMPan/checkpoints/BiMPan_WV3/BiMPan_WV3_2024_02_17-14_16_14.pth.tar'

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics = load_checkpoint(torch.load(
            checkpoint_path), model, optimizer, tr_metrics, val_metrics)

    # evaluation mode
    model.eval()
    with torch.no_grad():
        print("\n==> Start testing ...")
        test_progress_bar = tqdm(iter(test_loader), total=len(
            test_loader), desc="Testing", leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
        for pan, mslr, mshr in test_progress_bar:
            # forward
            mslr = torch.nn.functional.interpolate(
                mslr, size=(256, 256), mode='bilinear')

            pan, mslr, mshr = pan.to(device), mslr.to(device), mshr.to(device)
            mssr = model(pan, mslr)
            test_loss = criterion(mssr, mshr)
            test_metric = test_metric_collection.forward(mssr, mshr)
            test_report_loss += test_loss

            # report metrics
            test_progress_bar.set_postfix(
                loss=f'{test_loss.item()}', psnr=f'{test_metric["psnr"].item():.2f}', ssim=f'{test_metric["ssim"].item():.2f}')

        # compute metrics total
        test_report_loss = test_report_loss / len(test_loader)
        test_metric = test_metric_collection.compute()
        test_metrics.append({'loss': test_report_loss.item(),
                             'psnr': test_metric['psnr'].item(),
                             'ssim': test_metric['ssim'].item()})

        print(
            f'\nTesting: avg_loss = {test_report_loss.item():.4f} , avg_psnr= {test_metric["psnr"]:.4f}, avg_ssim={test_metric["ssim"]:.4f}')

        # reset metrics
        test_report_loss = 0
        test_metric_collection.reset()
        print("==> End testing <==\n")

        # train mode
        model.train()


if __name__ == '__main__':
    main()
