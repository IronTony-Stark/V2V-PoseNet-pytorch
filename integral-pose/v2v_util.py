import sys
import numpy as np


def discretize(coord, cropped_size):
    '''[-1, 1] -> [0, cropped_size]'''
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord - min_normalized) / scale 


def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    '''
    Map coordinates in set [0, 1, .., cropped_size-1] to original range [-cubic_size/2+refpoint, cubic_size/2 + refpoint]
    '''
    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size
    coord = coord * scale + min_normalized  # -> [-1, 1]

    coord = coord * cubic_size / 2 + refpoint

    return coord


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


def extract_coord_from_output(output, center=True):
    '''
    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
    center: if True, add 0.5, default is true
    return: shape (batch, jointNum, 3)
    '''
    assert(len(output.shape) >= 3)
    vsize = output.shape[-3:]

    output_rs = output.reshape(-1, np.prod(vsize))
    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
    max_index = np.array(max_index).T
    
    xyz_output = max_index.reshape([*output.shape[:-3], 3])

    # Note discrete coord can represents real range [coord, coord+1), see function scattering() 
    # So, move coord to range center for better fittness
    if center: xyz_output = xyz_output + 0.5

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

    # move cropped center to (virtual larger, [0, original_size]) original volume center
    # that is, treat current data as cropped from center of original volume, and now we put back it
    coord += (original_size / 2 - cropped_size / 2)

    # resize around original center with scale new_size/100
    resize_scale = 100 / new_size
    if new_size < 100:
        coord = coord * resize_scale + original_size/2 * (1 - resize_scale)
    elif new_size > 100:
        coord = coord * resize_scale - original_size/2 * (resize_scale - 1)
    else:
        # new_size = 100 if it is in test mode
        pass

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:,0] -= original_size/2
        original_coord[:,1] -= original_size/2
        coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
        coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle)
        coord[:,0] += original_size/2
        coord[:,1] += original_size/2

    # translation
    # Note, if trans = (original_size/2 - cropped_size/2 + 1), the following translation will
    # cancel the above translation(after discretion). It will be set it when in test mode. 
    # TODO: Can only achieve translation [-4, 4]?
    coord -= trans

    return coord


def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord, cropped_size)

    return cubic


# def generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
#     _, cropped_size, _ = sizes
#     d3output_x, d3output_y, d3output_z = d3outputs

#     coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
#     coord /= pool_factor  # [0, cropped_size/pool_factor]

#     # heatmap generation
#     output_size = int(cropped_size / pool_factor)
#     heatmap = np.zeros((keypoints.shape[0], output_size, output_size, output_size))

#     # use center of cell
#     center_offset = 0.5

#     for i in range(coord.shape[0]):
#         xi, yi, zi= coord[i]
#         heatmap[i] = np.exp(-(np.power((d3output_x+center_offset-xi)/std, 2)/2 + \
#             np.power((d3output_y+center_offset-yi)/std, 2)/2 + \
#             np.power((d3output_z+center_offset-zi)/std, 2)/2))  # +0.5, move coordinate to range center

#     return heatmap


#---
def containing_box_coord(point):
    '''
    point: (K, 3)
    return: (K, 8, 3), eight box vertices coords
    '''
    # if np.any(point >= 43):
    #     res = point[point >= 43]
    #     print('invalid point: {}'.format(res))
    #     exit()


    box_grid = np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij')
    box_grid = np.array(box_grid).reshape((3, 8)).transpose()

    floor = np.floor(point)
    box_coord = floor.reshape((-1, 1, 3)) + box_grid

    return box_coord


def box_coord_prob(point, box_coord):
    '''
    point: (K, 3)
    box_coord: (K, 8, 3)
    return: (K, 8)
    '''
    diff = box_coord - point.reshape((-1, 1, 3))
    weight = np.maximum(0, 1 - np.abs(diff))
    prob = weight[:,:,0] * weight[:,:,1] * weight[:,:,2]
    norm = np.sum(prob, axis=1, keepdims=True)
    norm[norm <= 0] = 1.0  # avoid zero
    prob = prob / norm

    return prob


def onehot_heatmap_impl(coord, output_size):
    '''
    coord: (K, 3)
    return: (output_size, output_size, output_size)
    '''
    coord = np.array(coord)

    box_coord = containing_box_coord(coord)  # (K, 8, 3)
    box_prob = box_coord_prob(coord, box_coord)  # (K, 8)
    box_coord = box_coord.astype(np.int32)

    # Generate K heatmaps
    heatmap = np.zeros((coord.shape[0], output_size, output_size, output_size))
    for i in range(coord.shape[0]):
        heatmap[i][box_coord[i,:,0], box_coord[i,:,1], box_coord[i,:,2]] = box_prob[i]

    return heatmap


def generate_onehot_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
    _, cropped_size, _ = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord /= pool_factor  # [0, cropped_size/pool_factor]

    # heatmap generation
    output_size = int(cropped_size / pool_factor)

    # Warning, clip joints into [0, 42], make sure the containing box coord indices will not exceed 43(44-1)
    # TODO: check here
    target_output_size = 44
    coord[coord >= target_output_size-2] = target_output_size - 2
    coord[coord < 0] = 0

    return onehot_heatmap_impl(coord, output_size)


def generate_volume_coord_gt(keypoints, refpoint, new_size, angle, trans, sizes, pool_factor):
    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord /= pool_factor

    # Warning, clip joints into [0, 42], make sure the containing box coord indices will not exceed 43(44-1)
    # TODO: check here
    target_output_size = 44
    coord[coord >= target_output_size-2] = target_output_size - 2
    coord[coord < 0] = 0

    return coord


class V2VVoxelization(object):
    def __init__(self, cubic_size, augmentation=True):
        self.cubic_size = cubic_size
        self.cropped_size, self.original_size = 88, 96
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
        trans = np.random.rand(3) * (self.original_size - self.cropped_size)

        if not self.augmentation:
            new_size = 100
            angle = 0
            trans = self.original_size/2 - self.cropped_size/2

        # Add noise and random selection
        add_noise = False
        random_selection = False

        if self.augmentation and add_noise:
            # noise, [-0.5, 0.5]
            scale = 0.5
            noise = (np.random.rand(*points.shape) * 2  - 1) * scale
            points += noise

        if self.augmentation and random_selection:
            threshold = np.random.rand(1)[0] * 0.5  # <= 0.5
            prob = np.random.rand(points.shape[0])
            mask = prob > threshold
            points = points[mask, :]

        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        # heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)
        # Use One-hot heatmap
        heatmap = generate_onehot_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)
        keypoints_volume_coords = generate_volume_coord_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.pool_factor)

        # one channel
        input = input.reshape((1, *input.shape))

        return input, heatmap, keypoints_volume_coords

    # def voxelize(self, points, refpoint):
    #     new_size, angle, trans = 100, 0, self.original_size/2 - self.cropped_size/2
    #     input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
    #     return input.reshape((1, *input.shape))

    # def generate_heatmap(self, keypoints, refpoint):
    #     new_size, angle, trans = 100, 0, self.original_size/2 - self.cropped_size/2
    #     heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)
    #     return heatmap

    # def evaluate(self, heatmaps, refpoints):
    #     coords = extract_coord_from_output(heatmaps, center=True)
    #     coords *= self.pool_factor
    #     keypoints = warp2continuous(coords, refpoints, self.cubic_size, self.cropped_size)
    #     return keypoints

    # def warp2continuous(self, coords, refpoints):
    #     print('Warning: added 0.5 on input coord')
    #     coords += 0.5  # move to grid cell center
    #     coords *= self.pool_factor
    #     keypoints = warp2continuous(coords, refpoints, self.cubic_size, self.cropped_size)
    #     return keypoints

    # def generate_coord_raw(self, points, refpoint):
    #     new_size, angle, trans = 100, 0, self.original_size/2 - self.cropped_size/2
    #     coord = generate_coord(points, refpoint, new_size, angle, trans, self.sizes)
    #     return coord

    def warp2continuous_raw(self, coords, refpoints):
        # Do not add 0.5, since coords have float precison
        coords = coords * self.pool_factor
        keypoints = warp2continuous(coords, refpoints, self.cubic_size, self.cropped_size)
        return keypoints