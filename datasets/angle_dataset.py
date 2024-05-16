import time

import numpy as np
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


# TODO: currently, one sample generation takes a stunning 10 seconds. Rotation is a the bottleneck.
#  SimpleITK's EulerTransform is much faster than scipy's rotate, but I wasn't able to make it work.
class AngleDataset(Dataset):
    def __init__(self, num_samples, size=256, dtype=np.uint8,
                 random_angle_low=10, random_angle_high=90,
                 random_noise_low=0, random_noise_high=80,
                 rect_intensity=255,
                 bright_voxel_removal_percentage=30,
                 transform=None):
        self.num_samples = num_samples
        self.size = np.array((size, size, size))
        self.dtype = dtype
        self.random_angle_low, self.random_angle_high = random_angle_low, random_angle_high
        self.random_noise_low, self.random_noise_high = random_noise_low, random_noise_high
        self.rect_intensity = rect_intensity
        self.bright_voxel_removal_percentage = bright_voxel_removal_percentage
        self.transform = transform

        self.rect_size = self.size[0] // 10
        self.volume_center = self.size // 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        """
        Generate a random volume that contains 2 connected parallelepipeds that have an angle between them.
        Apply random rotation, translation, and add noise to the volume.
        Return a volume, 14 keypoints (12 vertices of 2 parallelepipeds and 2 are their centers,
        and lastly the angle between the 2 parallelepipeds)
        """
        angle = np.random.randint(low=self.random_angle_low, high=self.random_angle_high)

        volume = np.zeros(self.size, dtype=self.dtype)

        # Create two adjacent parallelepipeds that have an angle between them
        rect1_size = np.array([self.rect_size, self.rect_size * 2, self.rect_size])
        rect1_loc = (self.size - rect1_size) // 2
        volume[
            rect1_loc[0]:rect1_loc[0] + rect1_size[0],
            rect1_loc[1]:rect1_loc[1] + rect1_size[1],
            rect1_loc[2]:rect1_loc[2] + rect1_size[2]
        ] = self.rect_intensity

        volume = rotate(volume, angle, axes=(0, 1), reshape=False, mode='nearest')  # z

        rect2_loc = np.copy(rect1_loc)
        rect2_loc[0] += self.rect_size  # raise the second rectangle, so that there is an angle
        rect2_loc[1] -= self.rect_size  # move the second rectangle to the right
        rect2_size = np.copy(rect1_size)
        rect2_size[1] += self.rect_size  # make the second rectangle longer
        volume[
            rect2_loc[0]:rect2_loc[0] + rect2_size[0],
            rect2_loc[1]:rect2_loc[1] + rect2_size[1],
            rect2_loc[2]:rect2_loc[2] + rect2_size[2]
        ] = self.rect_intensity

        # Randomly rotate the volume along every axis
        rotation_angles = np.random.randint(0, 360, size=3)
        volume = rotate(volume, rotation_angles[0], axes=(1, 2), reshape=False, mode='nearest')  # x
        # Invert the direction for y-axis rotation to stay consistent with scipy.spatial.transform.Rotation
        volume = rotate(volume, -rotation_angles[1], axes=(0, 2), reshape=False, mode='nearest')  # y
        volume = rotate(volume, rotation_angles[2], axes=(0, 1), reshape=False, mode='nearest')  # z

        # Randomly translate the volume along every axis
        translation = np.random.randint(-self.rect_size * 2, self.rect_size * 2, size=3, dtype=np.int8)
        volume = np.roll(volume, translation, axis=(0, 1, 2))

        # Randomly remove voxels
        # volume = remove_voxels(volume, 0, self.bright_voxel_removal_percentage)

        # Add random noise
        # noise = np.random.normal(self.random_noise_low, self.random_noise_high, self.size)
        # volume = np.clip(volume + noise, 0, 255).astype(self.dtype)

        # Extract vertices and centers of the parallelepipeds to create keypoints
        rect1_vertices = AngleDataset.get_parallelepipeds_vertices(rect1_loc, rect1_size)
        rect1_vertices = rotate_3d_points(rect1_vertices, (0, 0, angle), self.volume_center)
        rect1_vertices = rotate_3d_points(rect1_vertices, rotation_angles, self.volume_center)
        rect1_vertices += translation
        rect1_vertices = rect1_vertices.astype(self.dtype)

        rect2_vertices = AngleDataset.get_parallelepipeds_vertices(rect2_loc, rect2_size)
        rect2_vertices = rotate_3d_points(rect2_vertices, rotation_angles, self.volume_center)
        rect2_vertices += translation
        rect2_vertices = rect2_vertices.astype(self.dtype)

        keypoints = np.concatenate([rect1_vertices, rect2_vertices])

        # Debugging: draw the keypoints on the volume
        # for x, y, z in keypoints:
        #     volume[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = 255
        # for x, y, z in self._calculate_angle(keypoints):
        #     x, y, z = int(x), int(y), int(z)
        #     volume[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = 255
        # for x, y, z in [AngleDataset.calculate_parallelepipeds_translation(keypoints)]:
        #     x, y, z = int(x), int(y), int(z)
        #     volume[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = 255

        # Calculate the angle between the 2 parallelepipeds to ensure it matches the defined
        batch_keypoints = keypoints[np.newaxis, :]  # add batch dimension

        calc_angle = AngleDataset.calculate_parallelepipeds_angle(batch_keypoints)[0]
        if calc_angle > 90:
            calc_angle = 180 - calc_angle
        if abs(angle - calc_angle) > 3:
            print(f"[WARN]: Angle difference is too high: {angle} vs {calc_angle}")
        # assert abs(angle - calc_angle) < 3

        out = (volume, keypoints, AngleDataset.calculate_parallelepipeds_translation(batch_keypoints)[0],
               AngleDataset.calculate_parallelepipeds_orientation(batch_keypoints)[0], angle)
        if self.transform:
            return self.transform(*out)
        return out

    @staticmethod
    def get_parallelepipeds_vertices(loc, size):
        x, y, z = loc
        width, height, depth = size

        vertices = np.array([
            [x, y, z],
            [x + width, y, z],
            [x + width, y + height, z],
            [x, y + height, z],
            [x, y, z + depth],
            [x + width, y, z + depth],
            [x + width, y + height, z + depth],
            [x, y + height, z + depth],
            [(x + x + width) / 2, (y + y + height) / 2, (z + z + depth) / 2],  # Center
        ])

        return vertices

    @staticmethod
    def calculate_parallelepipeds_angle(keypoints_batch):
        """
        Calculate the angle between 2 parallelepipeds based on their keypoints
        :param keypoints_batch: [batch_size, num_keypoints, 3]
        """
        batch_size = keypoints_batch.shape[0]
        angles = np.zeros(batch_size)

        for i in range(batch_size):
            keypoints = keypoints_batch[i]

            top_front = np.mean(keypoints[[2, 3, 6, 7]], axis=0)
            top_back = np.mean(keypoints[[0, 1, 4, 5]], axis=0)
            bottom_front = np.mean(keypoints[[11, 12, 15, 16]], axis=0)
            bottom_back = np.mean(keypoints[[9, 10, 13, 14]], axis=0)

            # return [bottom_back, bottom_front, top_back, top_front]  # for debugging

            bottom_vector = bottom_front - bottom_back
            top_vector = top_front - top_back

            angles[i] = angle_between_vectors(bottom_vector, top_vector)

        return angles

    @staticmethod
    def calculate_parallelepipeds_translation(keypoints_batch):
        """
        Returns the coordinates of the base (bottom) parallelepiped's center
        :param keypoints_batch: [batch_size, num_keypoints, 3]
        """
        bottom_front = np.mean(keypoints_batch[:, [11, 12, 15, 16]], axis=1)
        bottom_back = np.mean(keypoints_batch[:, [9, 10, 13, 14]], axis=1)
        return (bottom_front + bottom_back) / 2

    @staticmethod
    def calculate_parallelepipeds_orientation(keypoints_batch):
        """
        Returns the orientation of the base (bottom) parallelepiped by calculating the normalized vector
        between the 'front' (near the top parallelepiped) and 'back' parts of the base
        :param keypoints_batch: [batch_size, num_keypoints, 3]
        """
        bottom_front = np.mean(keypoints_batch[:, [11, 12, 15, 16]], axis=1)
        bottom_back = np.mean(keypoints_batch[:, [9, 10, 13, 14]], axis=1)

        vector = bottom_front - bottom_back
        vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
        return vector


def rotate_3d_points(points, angles_degrees, center):
    """
    Rotate 3D points around the origin
    :param points: array of 3D points [n, 3]
    :param angles_degrees: angles in degrees along each axis [x, y, z]
    :param center: center of rotation. For 3D volumes, this is usually the center of the volume
    :return:
    """
    angles_radians = np.radians(angles_degrees)

    rotation_x = Rotation.from_euler('x', angles_radians[0])
    rotation_y = Rotation.from_euler('y', angles_radians[1])
    rotation_z = Rotation.from_euler('z', angles_radians[2])

    total_rotation = rotation_z * rotation_y * rotation_x

    # Translate to the origin, rotate, and translate back
    translated_points = points - center
    rotated_points = total_rotation.apply(translated_points)
    rotated_points += center

    return rotated_points


def remove_voxels(volume, threshold, remove_percentage=50):
    """
    Randomly remove (assign zero) a specified percentage of voxels that are above a certain threshold from the volume
    """
    target_voxels = (volume >= threshold)

    # Generate a random mask to remove a specified percentage of white voxels
    random_mask = np.random.choice([False, True], size=target_voxels.shape,
                                   p=[1 - remove_percentage / 100, remove_percentage / 100])

    volume[random_mask & target_voxels] = 0

    return volume


def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
        # raise ValueError("One or both vectors have zero length.")

    cosine_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

    angle_deg = np.degrees(angle_rad)

    return angle_deg


if __name__ == '__main__':
    ds = AngleDataset(1)

    start = time.time()
    volume, keypoints, _, _, angle = ds[0]
    print(f"Elapsed time: {time.time() - start} seconds")

    print(len(keypoints))

    import SimpleITK as sitk
    vol = sitk.GetImageFromArray(volume)
    sitk.Show(vol)
