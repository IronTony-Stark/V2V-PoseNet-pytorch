import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
import os
import platform
import wandb
from tqdm import tqdm

from lib.solver import train_epoch, val_epoch, test_epoch
from lib.sampler import ChunkSampler
from src.v2v_model import V2VModel
from src.v2v_util import V2VVoxelization
from datasets.msra_hand import MARAHandDataset
from lib.result_collector import BatchResultCollector
from lib.accuracy import compute_dist_acc_wrapper, compute_mean_err
import matplotlib.pyplot as plt
from vis.plot import plot_acc, plot_mean_err


#######################################################################################
# Note,
# Run in project root direcotry(ROOT_DIR) with:
# PYTHONPATH=./ python experiments/msra-subject3/main.py
# 
# This script will train model on MSRA hand datasets, save checkpoints to ROOT_DIR/checkpoint,
# and save test results(test_res.txt) and fit results(fit_res.txt) to ROOT_DIR.
#


#######################################################################################
## Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hand Keypoints Estimation Training')
    # parser.add_argument('--resume', 'r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume', '-r', default=-1, type=int, help='resume after epoch')
    args = parser.parse_args()
    return args


def log_epoch(wandb_run, pred_keypoints, gt_keypoints, subset_type='test', epoch=0):
    names = [f'joint{i + 1}' for i in range(keypoints_num)]

    dist, acc = compute_dist_acc_wrapper(pred_keypoints, gt_keypoints, max_dist=100, num=100)
    _, ax = plt.subplots()
    plot_acc(ax, dist, acc, names)
    wandb_run.log({f"{subset_type} Distance Accuracies Epoch {epoch}": ax})

    mean_err = compute_mean_err(pred_keypoints, gt_keypoints)
    _, ax = plt.subplots()
    plot_mean_err(ax, mean_err, names)
    wandb_run.log({f"{subset_type} Keypoint Errors Epoch {epoch}": ax})

    mean_err_all = compute_mean_err(pred_keypoints.reshape((-1, 1, 3)), gt_keypoints.reshape((-1, 1, 3)))
    wandb_run.log({f"{subset_type}/mean_error": mean_err_all})


#######################################################################################
## Configurations
# print('Warning: disable cudnn for batchnorm first, or just use only cuda instead!')

# When we need to resume training, enable randomness to avoid seeing the determinstic
# (agumented) samples many times.
# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

#
args = parse_args()
resume_train = args.resume >= 0
resume_after_epoch = args.resume

save_checkpoint = True
checkpoint_per_epochs = 1
checkpoint_dir = r'./checkpoints'

start_epoch = 0
epochs_num = 3

batch_size = 12

loader_num_workers = 6 if platform.system() != 'Windows' else 0

run = wandb.init(
    project="Pose Estimation using OCT ICRA 2025",
    config={
        "epochs": epochs_num,
        "batch_size": batch_size,
    },
    tags=["baseline", "V2V Pose Net cvpr15_MSRAHandGestureDB"],
    # mode="disabled",
)

#######################################################################################
## Data, transform, dataset and loader
# Data
print('==> Preparing data ..')

# data_dir = r'/gpfs/space/home/zaliznyi/data/cvpr15_MSRAHandGestureDB'
# center_dir = r'/gpfs/space/home/zaliznyi/projects/V2V-PoseNet-pytorch/datasets/msra_center'

data_dir = r'C:/Data/cvpr15_MSRAHandGestureDB'
center_dir = r'C:/Projects/V2V-PoseNet-pytorch/datasets/msra_center'

keypoints_num = 21
test_subject_id = 3
cubic_size = 200

# Transform
voxelization_train = V2VVoxelization(cubic_size=200, augmentation=True)
voxelization_val = V2VVoxelization(cubic_size=200, augmentation=False)


def transform_train(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert (keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return torch.from_numpy(input), torch.from_numpy(heatmap), {
        'refpoints': torch.from_numpy(refpoint.reshape((1, -1))),
        'joints': keypoints
    }


def transform_val(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert (keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return torch.from_numpy(input), torch.from_numpy(heatmap), {
        'refpoints': torch.from_numpy(refpoint.reshape((1, -1))),
        'joints': keypoints
    }


def transform_test(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert (keypoints.shape[0] == keypoints_num)
    input = voxelization_train.voxelize(points, refpoint)
    return torch.from_numpy(input), {
        'refpoints': torch.from_numpy(refpoint.reshape((1, -1))),
        'joints': keypoints
    }


def transform_output(heatmaps, refpoints):
    keypoints = voxelization_train.evaluate(heatmaps, refpoints)
    return keypoints


# Dataset and loader
print(f'==> Preparing dataloaders with {loader_num_workers} workers ..')

train_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)
train_res_collector = BatchResultCollector(len(train_set), transform_output)

# No separate validation dataset, just use test dataset instead
val_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_val)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers)
val_res_collector = BatchResultCollector(len(val_set), transform_output)

#######################################################################################
## Model, criterion and optimizer
print('==> Constructing model ..')
net = V2VModel(input_channels=1, output_channels=keypoints_num)

net = net.to(device, dtype)
if device == torch.device('cuda'):
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
print('cudnn.enabled: ', torch.backends.cudnn.enabled)

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters())
# optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)


#######################################################################################
## Resume
if resume_train:
    # Load checkpoint
    epoch = resume_after_epoch
    checkpoint_file = os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pth')

    print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

#######################################################################################
## Train and Validate
print('==> Training ..')
for epoch in tqdm(range(start_epoch, start_epoch + epochs_num), desc="Epochs"):
    train_epoch(net, criterion, optimizer, train_loader, train_res_collector, device=device, dtype=dtype, wandb_run=run)
    log_epoch(run, train_res_collector.get_pred_keypoints(), train_res_collector.get_gt_keypoints(),
              subset_type='train', epoch=epoch)
    train_res_collector.reset()

    val_epoch(net, criterion, val_loader, val_res_collector, device=device, dtype=dtype, wandb_run=run)
    log_epoch(run, val_res_collector.get_pred_keypoints(), val_res_collector.get_gt_keypoints(),
              subset_type='val', epoch=epoch)
    val_res_collector.reset()

    if save_checkpoint and epoch % checkpoint_per_epochs == 0:
        if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch' + str(epoch) + '.pth')
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_file)

#######################################################################################
## Test
print('==> Testing ..')

print('Test on test dataset ..')
test_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers)
test_res_collector = BatchResultCollector(len(test_set), transform_output)

test_epoch(net, test_loader, test_res_collector, device, dtype, run)
log_epoch(run, test_res_collector.get_pred_keypoints(), test_res_collector.get_gt_keypoints(), subset_type='test')

print('Fit on train dataset ..')
fit_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_test)
fit_loader = DataLoader(fit_set, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers)
fit_res_collector = BatchResultCollector(len(fit_set), transform_output)

test_epoch(net, fit_loader, fit_res_collector, device, dtype)
log_epoch(run, fit_res_collector.get_pred_keypoints(), fit_res_collector.get_gt_keypoints(), subset_type='fit')

print('All done ..')
