import argparse
import os
import platform
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.angle_dataset import AngleDataset
from lib.solver import train_epoch
from src.v2v_model import V2VModel
from src.v2v_util import V2VVoxelization, extract_coord_from_output


# TODO add weights monitoring to mitigate exploding/vanishing gradients
#######################################################################################
# Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hand Keypoints Estimation Training')
    # parser.add_argument('--resume', 'r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume', '-r', default=-1, type=int, help='resume after epoch')
    args = parser.parse_args()
    return args


# class CombinedToyDataLoss(nn.Module):
#     def __init__(self, pool_factor=2, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.mse = nn.MSELoss()
#         self.l1 = nn.L1Loss()
#         self.cosine = nn.CosineSimilarity()
#
#         self.pool_factor = pool_factor
#
#     # TODO: a lot of this has to be modified to support gradient flow.
#     #  For one, voxel coordinates calculation should be done using differentiable_argmax
#     def forward(self, outputs, targets):
#         keypoints = extract_coord_from_output(outputs)
#         keypoints *= self.pool_factor
#
#         target_heatmap, target_translation, target_orientation, target_angle = \
#             targets['heatmap'], targets['translation'], targets['orientation'], targets['angle']
#
#         translation = AngleDataset.calculate_parallelepipeds_translation(keypoints)
#         orientation = AngleDataset.calculate_parallelepipeds_orientation(keypoints)
#         angle = AngleDataset.calculate_parallelepipeds_angle(keypoints)
#
#         keypoint_loss = self.mse(outputs, target_heatmap)
#         translation_loss = self.l1(translation, target_translation)
#         orientation_loss = self.cosine(orientation, target_orientation)
#         angle_loss = self.mse(angle, target_angle)
#
#         return keypoint_loss + translation_loss + orientation_loss + angle_loss


def generate_heatmap(keypoint_coordinates, pool_factor, output_size, std):
    """
    Generate heatmap from keypoint coordinates
    :param keypoint_coordinates: [N, 3] array of keypoint coordinates
    :param pool_factor: int determines how output heatmap size is different from the input volume size
    (e.g. if volume is 88x88x88 and heatmap is 44x44x44, then pool_factor is 2)
    :param output_size: the size of the output heatmap
    :param std: used in the Gaussian function to control the spread of the heatmap around each keypoint.
    A smaller std value will result in a more concentrated heatmap around the keypoints, while
    a larger std value will spread the heatmap more widely.
    :return:
    """
    keypoint_coordinates /= pool_factor

    heatmap = np.zeros((keypoint_coordinates.shape[0], output_size, output_size, output_size))

    d3output_x, d3output_y, d3output_z = np.meshgrid(
        np.arange(output_size), np.arange(output_size), np.arange(output_size), indexing='ij')

    center_offset = 0.5  # use center of cell
    for i in range(keypoint_coordinates.shape[0]):
        xi, yi, zi = keypoint_coordinates[i]
        heatmap[i] = np.exp(-(
            np.power((d3output_x + center_offset - xi) / std, 2) / 2 +
            np.power((d3output_y + center_offset - yi) / std, 2) / 2 +
            np.power((d3output_z + center_offset - zi) / std, 2) / 2
        ))

    return heatmap


#######################################################################################
# Configurations
@dataclass
class Config:
    keypoints = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float

    args = parse_args()
    resume_train = args.resume >= 0
    resume_after_epoch = args.resume

    save_checkpoint = True
    checkpoint_per_epochs = 5
    checkpoint_dir = r'./checkpoints/keypoints' if keypoints else r'./checkpoints/regression'

    start_epoch = resume_after_epoch if resume_train else 0
    epochs_num = 100

    batch_size = 12

    loader_num_workers = 6 if platform.system() != 'Windows' else 0

    output_channels = 18 if keypoints else 7


config = Config()

run = wandb.init(
    project="Pose Estimation using OCT ICRA 2025",
    name=f"{'Keypoints' if config.keypoints else 'Regression'}",
    config=asdict(config),
    tags=["toy-dataset", "V2V Pose Net cvpr15_MSRAHandGestureDB"],
    mode="disabled",
)

#######################################################################################
# Data, transform, dataset and loader

# Data
print('==> Preparing data ..')

if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)


# Transform
def transform_keypoints(volume, keypoints, translation, orientation, angle):
    volume = volume / 255.0
    volume = volume[np.newaxis, :]

    target = generate_heatmap(keypoints.astype(np.float64), pool_factor=2, output_size=44, std=1.7)

    extra = {
        'translation': translation,
        'orientation': orientation,
        'angle': angle
    }

    return torch.from_numpy(volume), torch.from_numpy(target), extra


def transform_regression(volume, keypoints, translation, orientation, angle):
    volume = volume / 255.0
    volume = volume[np.newaxis, :]

    target = np.zeros(7, dtype=np.float64)
    target[:3] = translation
    target[3:6] = orientation
    target[6] = angle

    extra = {
        'translation': translation,
        'orientation': orientation,
        'angle': angle
    }

    return torch.from_numpy(volume), torch.from_numpy(target), extra


# Dataset and loader
print(f'==> Preparing dataloaders with {config.loader_num_workers} workers ..')

train_set = AngleDataset(num_samples=400, size=88,
                         transform=transform_keypoints if config.keypoints else transform_regression)
train_loader = DataLoader(train_set, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.loader_num_workers)

#######################################################################################
# Model, criterion and optimizer
print('==> Constructing model ..')

net = V2VModel(input_channels=1, output_channels=config.output_channels, keypoints=config.keypoints, volume_size=88.0)
net = net.to(config.device, config.dtype)
if config.device == torch.device('cuda'):
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
print('cudnn.enabled: ', torch.backends.cudnn.enabled)

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters())  # TODO LR decay

#######################################################################################
# Resume
if config.resume_train:
    # Load checkpoint
    epoch = config.resume_after_epoch
    checkpoint_file = os.path.join(config.checkpoint_dir, f'epoch{epoch}.pth')

    print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
    assert os.path.isdir(config.checkpoint_dir), 'Error: no checkpoint directory found!'
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

    checkpoint = torch.load(os.path.join(config.checkpoint_dir, f'epoch{epoch}.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    config.start_epoch = checkpoint['epoch'] + 1

#######################################################################################
# Train and Validate
print('==> Training ..')


class MetricsCalcKeypoints:
    def __init__(self):
        pass

    def __call__(self, outputs, targets, extras):
        target_translation, target_orientation, target_angle = \
            extras['translation'], extras['orientation'], extras['angle']
        target_translation, target_orientation, target_angle = target_translation.detach().cpu().numpy(), \
            target_orientation.detach().cpu().numpy(), target_angle.detach().cpu().numpy()

        outputs = outputs.detach().cpu().numpy()
        keypoints = extract_coord_from_output(outputs)
        keypoints *= 2  # pool factor

        translation = AngleDataset.calculate_parallelepipeds_translation(keypoints)
        orientation = AngleDataset.calculate_parallelepipeds_orientation(keypoints)
        angle = AngleDataset.calculate_parallelepipeds_angle(keypoints)
        angle[angle > 90] = 180 - angle[angle > 90]

        # Calculate avg distance error between translation (position) and target_translation (target position)
        translation_avg_dist_error = np.mean(np.linalg.norm(translation - target_translation, axis=1))
        orientation_avg_error = np.mean(np.linalg.norm(orientation - target_orientation, axis=1))
        angle_avg_error = np.mean(np.abs(angle - target_angle))

        return {
            "translation_avg_dist_error": translation_avg_dist_error,
            "orientation_avg_error": orientation_avg_error,
            "angle_avg_error": angle_avg_error,
        }


class MetricsCalcRegression:
    def __init__(self):
        pass

    def __call__(self, outputs, targets, extras):
        target_translation, target_orientation, target_angle = \
            extras['translation'], extras['orientation'], extras['angle']
        target_translation, target_orientation, target_angle = target_translation.detach().cpu().numpy(), \
            target_orientation.detach().cpu().numpy(), target_angle.detach().cpu().numpy()

        outputs = outputs.detach().cpu().numpy()
        translation = outputs[:, :3]
        orientation = outputs[:, 3:6]
        angle = outputs[:, 6]

        # Calculate avg distance error between translation (position) and target_translation (target position)
        translation_avg_dist_error = np.mean(np.linalg.norm(translation - target_translation, axis=1))
        orientation_avg_error = np.mean(np.linalg.norm(orientation - target_orientation, axis=1))
        angle_avg_error = np.mean(np.abs(angle - target_angle))

        return {
            "translation_avg_dist_error": translation_avg_dist_error,
            "orientation_avg_error": orientation_avg_error,
            "angle_avg_error": angle_avg_error,
        }


metrics_calc = MetricsCalcKeypoints() if config.keypoints else MetricsCalcRegression()
for epoch in tqdm(range(config.start_epoch, config.start_epoch + config.epochs_num), desc="Epochs"):
    train_epoch(net, criterion, optimizer, train_loader, epoch, device=config.device, dtype=config.dtype,
                wandb_run=run, metrics_calc=metrics_calc)

    if config.save_checkpoint and epoch % config.checkpoint_per_epochs == 0:
        checkpoint_file = os.path.join(config.checkpoint_dir, f'epoch{epoch}.pth')
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_file)
