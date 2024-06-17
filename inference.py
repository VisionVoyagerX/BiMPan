from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis

from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from data_loader.DataLoader import DIV2K, GaoFen2, Sev2Mod, WV3, GaoFen2panformer
from BiMPan import BiMPan
from utils import *
import time
import numpy as np


def measure_gpu_throughput(model, input1_, input2_):
    input1 = input1_.to('cuda')
    input2 = input2_.to('cuda')
    model = model.to('cuda')

    ave_forward_throughput = []

    ave_start = time.time()
    for t in range(300):
        start = time.time()
        x = model(input1, input2)
        end = time.time()
        fwd_throughput = 1/(end-start)
        # print('forward_throughput is {:.4f}'.format(fwd_throughput))
        ave_forward_throughput.append(fwd_throughput)

    ave_fwd_throughput = np.mean(ave_forward_throughput[1:])

    print('Mean throughput over 300 runs: {:.4f}'.format(ave_fwd_throughput))


def measure_gpu_latency(model, input1_, input2_):
    input1 = input1_.to('cuda')
    input2 = input2_.to('cuda')
    model = model.to('cuda')

    repetitions = 300

    # GPU-WARM-UP
    for _ in range(20):  # Increase warm-up iterations to ensure GPU is fully warmed up
        _ = model(input1, input2)
        torch.cuda.synchronize()  # Ensure the warm-up operation completes

    # Measure performance
    timings = []
    with torch.no_grad():
        for _ in range(repetitions):
            start = time.time()
            _ = model(input1, input2)
            torch.cuda.synchronize()  # Ensure the operation completes
            end = time.time()
            latency = end - start
            timings.append(latency)

    mean_latency = np.mean(timings)
    print(
        f"Mean time over {repetitions} runs: {mean_latency} seconds")
    return mean_latency


def main():
    # Prepare device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)

    choose_dataset = 'WV3'  # choose dataset

    if choose_dataset == 'GaoFen2':
        dataset = eval('GaoFen2')
        tr_dir = 'data/pansharpenning_dataset/GF2/train/train_gf2.h5'
        eval_dir = 'data/pansharpenning_dataset/GF2/val/valid_gf2.h5'
        test_dir = 'data/pansharpenning_dataset/GF2/test/test_gf2_multiExm1.h5'
        checkpoint_dir = 'checkpoints/BiMPan_GF2/BiMPan_GF2_2024_02_18-12_21_11.pth.tar'
        ms_channel = 4
        ergas_l = 4
    elif choose_dataset == 'WV3':
        dataset = eval('WV3')
        tr_dir = 'data/pansharpenning_dataset/WV3/train/train_wv3.h5'
        eval_dir = 'data/pansharpenning_dataset/WV3/val/valid_wv3.h5'
        test_dir = 'data/pansharpenning_dataset/WV3/test/test_wv3_multiExm1.h5'
        checkpoint_dir = 'checkpoints/BiMPan_WV3/BiMPan_WV3_2024_02_17-14_16_14.pth.tar'
        ms_channel = 8
        ergas_l = 4
    else:
        print(choose_dataset, ' does not exist')

    # Initialize DataLoader
    train_dataset = dataset(
        Path(tr_dir), transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])  # /home/ubuntu/project
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, drop_last=True)

    # validation_dataset = dataset(
    #   Path(eval_dir))
    # validation_loader = DataLoader(
    #    dataset=validation_dataset, batch_size=64, shuffle=True)

    test_dataset = dataset(
        Path(test_dir))
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    model = BiMPan(ms_channels=ms_channel, mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                   pan_std=train_dataset.pan_std.to(device)).to(device)

    optimizer = Adam(model.parameters(), lr=0.0003,
                     betas=(0.9, 0.999))

    criterion = MSELoss().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
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
    """pan_example = torch.randn(
        (1, 1, 256, 256)).to(device)
    mslr_example = torch.randn(
        (1, ms_channel, 64 * 4, 64 * 4)).to(device)"""

    # summary(model, pan_example, mslr_example, verbose=1)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    lr_decay_intervals = 50000

    continue_from_checkpoint = True

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics = load_checkpoint(torch.load(
            checkpoint_dir), model, optimizer, tr_metrics, val_metrics)

    # input_tensor1 = torch.randn(
    #    1, 1, 256, 256).to(device)  # Example input tensor 1
    # input_tensor2 = torch.randn(
    #    1, ms_channel, 256, 256).to(device)  # Example input tensor 2

    # measure_gpu_throughput(model, input_tensor1, input_tensor2)
    # measure_gpu_latency(model, input_tensor1, input_tensor2)

    def scaleMinMax(x):
        return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))

    # evaluation mode
    model.eval()
    with torch.no_grad():
        print("\n==> Start testing ...")
        test_progress_bar = tqdm(iter(test_loader), total=len(
            test_loader), desc="Testing", leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
        for i, (pan, mslr, mshr) in enumerate(test_progress_bar):
            # forward
            mslr = torch.nn.functional.interpolate(
                mslr, size=(256, 256), mode='bilinear')

            pan, mslr, mshr = pan.to(device), mslr.to(device), mshr.to(device)
            mssr = model(pan, mslr)
            test_loss = criterion(mssr, mshr)
            test_metric = test_metric_collection.forward(mssr, mshr)
            test_report_loss += test_loss

            figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
            axis[0].imshow((scaleMinMax(mslr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[0].set_title('(a) LR')
            axis[0].axis("off")

            axis[1].imshow(pan.permute(0, 3, 2, 1).detach().cpu()[
                0, ...], cmap='gray')
            axis[1].set_title('(b) PAN')
            axis[1].axis("off")

            axis[2].imshow((scaleMinMax(mssr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[2].set_title(
                f'(c) GPPNN {test_metric["psnr"]:.2f}dB/{test_metric["ssim"]:.4f}')
            axis[2].axis("off")

            axis[3].imshow((scaleMinMax(mshr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[3].set_title('(d) GT')
            axis[3].axis("off")

            plt.savefig(f'results/Images_{choose_dataset}_{i}.png')

            mslr = mslr.permute(0, 3, 2, 1).detach().cpu().numpy()
            pan = pan.permute(0, 3, 2, 1).detach().cpu().numpy()
            mssr = mssr.permute(0, 3, 2, 1).detach().cpu().numpy()
            gt = mshr.permute(0, 3, 2, 1).detach().cpu().numpy()

            np.savez(f'results/img_array_{choose_dataset}_{i}.npz', mslr=mslr,
                     pan=pan, mssr=mssr, gt=gt)

        # compute metrics
        test_metric = test_metric_collection.compute()
        test_metric_collection.reset()

        # Print final scores
        print(f"Final scores:\n"
              f"ERGAS: {test_metric['ergas'].item()}\n"
              f"SAM: {test_metric['sam'].item()}\n"
              f"PSNR: {test_metric['psnr'].item()}\n"
              f"SSIM: {test_metric['ssim'].item()}")


if __name__ == '__main__':
    main()
