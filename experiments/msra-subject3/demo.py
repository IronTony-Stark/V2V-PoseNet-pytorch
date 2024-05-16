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
from src.v2v_util import scattering
import SimpleITK as sitk
from torchsummary import summary


class BatchResultCollector():
    def __init__(self, samples_num, transform_output):
        self.samples_num = 1
        self.transform_output = transform_output
        self.keypoints = None
        self.idx = 0

    def __call__(self, data_batch):
        inputs_batch, outputs_batch, extra_batch = data_batch
        outputs_batch = outputs_batch.cpu().numpy()
        refpoints_batch = extra_batch.cpu().numpy()

        keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

        if self.keypoints is None:
            # Initialize keypoints until dimensions awailable now
            self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

        batch_size = keypoints_batch.shape[0]
        self.keypoints[self.idx:self.idx + batch_size] = keypoints_batch
        self.idx += batch_size

    def get_result(self):
        return self.keypoints


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

# data_dir = r'/gpfs/space/home/zaliznyi/data/cvpr15_MSRAHandGestureDB'
# center_dir = r'/gpfs/space/home/zaliznyi/projects/V2V-PoseNet-pytorch/datasets/msra_center'

data_dir = r'C:/Data/cvpr15_MSRAHandGestureDB'
center_dir = r'C:/Projects/V2V-PoseNet-pytorch/datasets/msra_center'

keypoints = False
keypoints_num = 18 if keypoints else 7
test_subject_id = 3
cubic_size = 200

batch_size = 12

loader_num_workers = 6 if platform.system() != 'Windows' else 0

# Transform
voxelization_train = V2VVoxelization(cubic_size=200, augmentation=True)

def transform_test(sample):
    points, refpoint = sample['points'], sample['refpoint']
    input = voxelization_train.voxelize(points, refpoint)
    return torch.from_numpy(input), torch.from_numpy(refpoint.reshape((1, -1)))


def transform_output(heatmaps, refpoints):
    return voxelization_train.get_voxel_coordinates(heatmaps)

test_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers)
test_res_collector = BatchResultCollector(len(test_set), transform_output)

net = V2VModel(input_channels=1, output_channels=keypoints_num, keypoints=keypoints)

net = net.to(device, dtype)
if device == torch.device('cuda'):
    torch.backends.cudnn.enabled = False
    # cudnn.benchmark = True
print('cudnn.enabled: ', torch.backends.cudnn.enabled)

# checkpoints = torch.load('C:/Projects/V2V-PoseNet-pytorch/checkpoints/epoch8.pth')
# net.load_state_dict(checkpoints['model_state_dict'])

net.eval()

summary(net, (1, 88, 88, 88))

# with torch.no_grad():
#     inputs, extra = test_set[0]
#     inputs, extra = inputs.unsqueeze_(0), extra.unsqueeze_(0)
#     outputs = net(inputs.to(device, dtype))
#     test_res_collector((inputs, outputs, extra))
#
# volume = inputs.cpu().numpy()[0][0]
# keypoints = test_res_collector.get_result()[0]
# keypoint_volume = scattering(keypoints, volume.shape[0])
#
# volume = (volume * 255).astype(np.uint8)
# keypoint_volume = (keypoint_volume * 255).astype(np.uint8)
#
# white_voxel_coords = np.argwhere(keypoint_volume == 255)
# for i in range(white_voxel_coords.shape[0]):
#     x = white_voxel_coords[i, 0]
#     y = white_voxel_coords[i, 1]
#     z = white_voxel_coords[i, 2]
#
#     # For better visibility, make neighbouring voxels white as well
#     volume[x-1:x+2, y-1:y+2, z-1:z+2] = 255
#
# vol = sitk.GetImageFromArray(volume)
# sitk.Show(vol)
