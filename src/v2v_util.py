import sys
import numpy as np


# TODO:
# 1. check class PointsVoxelization
# 2. np or torch?
# 3. check resize



def discretize(coord, cropped_size):
    '''[-1, 1] -> [0, cropped_size]'''
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord - min_normalized) / scale 


def scattering(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    coord = coord.astype(np.int32)

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap 
    cubic[coord[:, 0], coord[:, 1], coord[:, 2]] = 1

    return cubic


def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    '''
    Map coordinates in set [0, 1, .., cropped_size-1] to range [-cropped_size/2+refpoint, cropped_size/2 + refpoint]
    '''
    # Note discrete coord can represents real range [coord, coord+1), see function scattering() 
    # So, move coord to range center for better fittness
    coord += 0.5

    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size
    coord = coord * scale + min_normalized  # -> [-1, 1]

    coord = coord * cubic_size / 2 + refpoint

    return coord


def extract_coord_from_output(output):
    '''
    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
    return: shape (batch, jointNum, 3)
    '''
    assert(len(output.shape) >= 3)
    vsize = output.shape[-3:]

    output_rs = output.reshape(-1, np.prod(vsize))
    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
    max_index = np.array(max_index).T
    
    xyz_output = max_index.reshape([*output.shape[:-3], 3])

    return xyz_output


def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes

    # points shape: (n, 3)
    coord = points

    # note, will consider points within range [refpoint-cubic_size/2, refpoint+cubic_size/2] as candidates

    # normalize
    coord = (coord - refpoint) / (cubic_size/2)  # -> [-1, 1]

    # discretize
    coord = discretize(coord, cropped_size)  # -> [0, cropped_size]
    coord += (original_size / 2 - cropped_size / 2) 

    # resize
    if new_size < 100:
        coord = coord / original_size * np.floor(original_size*new_size/100) + \
                np.floor(original_size/2 - original_size/2*new_size/100)
    elif new_size > 100:
        coord = coord / original_size * np.floor(original_size*new_size/100) - \
                np.floor(original_size/2*new_size/100 - original_size/2)
    else:
        # new_size = 100 if it is in test mode
        pass

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:,1] = original_size-1 - original_coord[:,1]
        original_coord[:,0] -= (original_size-1)/2
        original_coord[:,1] -= (original_size-1)/2
        coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
        coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle)
        coord[:,0] += (original_size-1)/2
        coord[:,1] += (original_size-1)/2
        coord[:,1] = original_size-1 - coord[:,1]

    # translation
    # Note, if trans = (original_size/2 - cropped_size/2), the following translation will
    # cancel the above translation(after discretion). It will be set it when in test mode. 
    coord -= trans - 1

    return coord


def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord, cropped_size)

    return cubic


def generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
    _, cropped_size, _ = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord /= pool_factor  # [0, cropped_size/pool_factor]

    # heatmap generation
    output_size = int(cropped_size / pool_factor)
    heatmap = np.zeros((keypoints.shape[0], output_size, output_size, output_size))

    for i in range(coord.shape[0]):
        xi, yi, zi= coord[i]
        heatmap[i] = np.exp(-(np.power((d3output_x+0.5-xi)/std, 2)/2 + \
            np.power((d3output_y+0.5-yi)/std, 2)/2 + \
            np.power((d3output_z+0.5-zi)/std, 2)/2))  # +0.5, move coordinate to range center

    return heatmap


class V2VVoxelization(object):
    def __init__(self, augmentation=True):
        self.cubic_size, self.cropped_size, self.original_size = 140, 88, 96
        self.sizes = (self.cubic_size, self.cropped_size, self.original_size)
        self.pool_factor = 2
        self.std = 1.7
        self.augmentation = augmentation

        output_size = int(self.cropped_size / self.pool_factor)
        # Note, range(size) and indexing = 'ij'
        self.d3outputs = np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size), indexing='ij')

    def __call__(self, sample):
        points, keypoints, refpoint = sample['points'], sample['keypoints'], sample['refpoint']

        ## Augmentations
        # Resize
        new_size = np.random.rand() * 40 + 80

        # Rotation
        angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi

        # Translation
        trans = np.random.randint(1, self.original_size-self.cropped_size+1+1, size=3)

        if not self.augmentation:
            new_size = 100
            angle = 0
            trans = self.original_size/2 - self.cropped_size/2 + 1

        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)

        return input.reshape((1, *input.shape)), heatmap